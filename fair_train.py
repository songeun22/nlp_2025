import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from transformers import TrainerCallback
from preprocess.tools import toxic_preprocess
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from huggingface_hub import login
from huggingface_hub import create_repo


class FairnessTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_hidden_states= True # True
        )
        ce_loss = outputs.loss[0]
        # print(ce_loss, ce_loss.shape) 
        # print(ce_loss[0])
        cls_emb = outputs.hidden_states[-1][:, 0, :]
        # cls_emb = outputs.last_hidden_state[:, 0, :]  

        # toxic direction
        v = self.v_tox.to(cls_emb.device)
        v = v / (v.norm() + 1e-8)
        proj = (cls_emb * v).sum(-1)

        # subgroup weights
        w = inputs["weights"]   # [B, K]
        K = int(w.shape[1])

        fair_total = 0.0
        fair_subgroup = torch.zeros(K)
        for k in range(K):
            w_k = w[:, k]
            m_k = (w_k * proj).sum() / (w_k.sum() + 1e-8)
            fair_subgroup[k] = m_k
            fair_total += torch.abs(m_k)

        fair_mean = torch.mean(fair_subgroup)
        loss = ce_loss + 0.3 * fair_total / K
        # loss = ce_loss + 0.3 * torch.mean(torch.pow(fair_subgroup - fair_mean, 2))

        self.log({"fairness_sum": fair_total.detach().cpu().item()})
        self.log({"ce_loss": ce_loss.detach().cpu().item()})
        self.log({"total_loss": loss.detach().cpu().item()})

        return (loss, outputs) if return_outputs else loss





class ToxicDirectionCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        trainer = self.trainer
        model = trainer.model.eval()
        device = trainer.model.device

        import random
        random_idx = random.sample(range(len(trainer.train_dataset)), 2000)

        # 샘플만 DataLoader로 만들기
        sample_subset = torch.utils.data.Subset(trainer.train_dataset, random_idx)
        sample_loader = DataLoader(
            sample_subset,
            batch_size=trainer.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=trainer.data_collator,
        )

        all_cls = []
        all_labels = []

        for batch in sample_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                )
                cls = outputs.hidden_states[-1][:, 0, :].cpu()

            all_cls.append(cls)
            all_labels.append(batch["labels"].cpu())

        all_cls = torch.cat(all_cls)
        all_labels = torch.cat(all_labels)

        mu_tox = all_cls[all_labels == 1].mean(0)
        mu_non = all_cls[all_labels == 0].mean(0)
        v = mu_tox - mu_non

        trainer.v_tox = v / (v.norm() + 1e-8)

        print(f"[Epoch {state.epoch}] v_tox updated. norm={trainer.v_tox.norm():.4f}. peek = {trainer.v_tox[:10]}")




def compute_directions(model, full_loader, device):
    """
    full_loader
    returns: v_tox, v_non, subgroup_means(dict)
    """
    model.eval()
    all_cls = []
    all_labels = []
    all_groups_w = [] # [N, K]

    with torch.no_grad():
        for batch in full_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)              # toxic/non-toxic
            groups = batch["groups"].to(device)              # subgroup indicator

            outputs = model(input_ids=input_ids, 
                            attention_mask=attn,
                            output_hidden_states=True)

            cls_emb = outputs.hidden_states[-1][:, 0, :]     # [B, 768]
            
            all_cls.append(cls_emb)
            all_labels.append(labels)
            all_groups.append(groups)

    all_cls = torch.cat(all_cls, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_groups = torch.cat(all_groups, dim=0)

    # toxic / non-toxic embeddings
    tox_emb = all_cls[all_labels == 1]
    non_emb = all_cls[all_labels == 0]

    mu_tox = tox_emb.mean(dim=0)
    mu_non = non_emb.mean(dim=0)

    v_tox = (mu_tox - mu_non)
    v_tox = v_tox / (v_tox.norm() + 1e-8)

    subgroup_ids = torch.unique(all_groups)
    subgroup_means = {}

    for gid in subgroup_ids:
        group_emb = all_cls[all_groups == gid]
        if len(group_emb) > 0:
            gmu = group_emb.mean(dim=0)
            subgroup_means[int(gid.item())] = gmu / (gmu.norm() + 1e-8)

    return v_tox.detach(), subgroup_means




def basic_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }




class FairToxic:

    def __init__(self, model_path, dataset):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, 
            num_labels = 2, 
            output_hidden_states = True 
        ).to(self.device)

        self.hidden_size = self.model.config.hidden_size

        self.dataset = dataset


    def collate_fn(self, batch):
            texts = [b["text"] for b in batch]
            labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
            subgroup = torch.tensor([b["weights"] for b in batch], dtype=torch.float32)

            # tokenizer 사용 (padding=True or max_length)
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            batch_out = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": labels,
                "weights": subgroup
            }
            return batch_out

    
    def prepare_dataset(self, dataset, batch_size): 

        hf_ds = Dataset.from_list(dataset)

        ds = hf_ds.train_test_split(test_size=0.1, seed=2025)
        train_ds = ds["train"]
        valid_ds = ds["test"]


        def tokenize_fn(batch):
            encoded = self.tokenizer(
                batch["text"],
                truncation=True,
                padding=False  # pad는 collator에서 수행
            )
            batch["input_ids"] = encoded["input_ids"]
            batch["attention_mask"] = encoded["attention_mask"]
            return batch

        train_ds = train_ds.map(tokenize_fn, batched=True)
        valid_ds = valid_ds.map(tokenize_fn, batched=True)

        def convert_subgroup(batch):
            batch["weights"] = [torch.tensor(w, dtype=torch.float32) for w in batch["weights"]]
            return batch

        train_ds = train_ds.map(convert_subgroup, batched=True)
        valid_ds = valid_ds.map(convert_subgroup, batched=True)

        columns = ["input_ids", "attention_mask", "label", "weights"]
        train_ds = train_ds.with_format("torch", columns=columns)
        valid_ds = valid_ds.with_format("torch", columns=columns)


        # print("\n====== Dataset Schema Check ======")
        # print("Train features:", train_ds.features)
        # print("Valid features:", valid_ds.features)


        sample = train_ds[0]
        # print("Train sample keys:", sample.keys())
        for key in columns:
            if key not in sample:
                print(f"❌ Missing key in sample: {key}")
            else:
                print(f"OK key: {key}, shape = {sample[key].shape if hasattr(sample[key], 'shape') else type(sample[key])}")

        print("=================================\n")

        return train_ds, valid_ds
    




    # def evaluate(self, trainer, ds, save_path):

    #     model = trainer.model
    #     device = model.device

    #     all_logits = []
    #     all_labels = []
    #     all_groups = []
    #     all_cls = []

    #     # ds = DataLoader(ds, batch_size= 16, shuffle=False, collate_fn = self.collate_fn)

    #     for batch in ds:
    #         batch = {k: v.to(device) for k, v in batch.items()}

    #         with torch.no_grad():
    #             outputs = model(
    #                 input_ids=batch["input_ids"],
    #                 attention_mask=batch["attention_mask"],
    #                 output_hidden_states=True
    #             )
    #         logits = outputs.logits
    #         cls_emb = outputs.hidden_states[-1][:, 0, :]

    #         all_logits.append(logits)
    #         all_labels.append(batch["labels"])
    #         all_groups.append(batch["weights"])
    #         all_cls.append(cls_emb)

    #     logits = torch.cat(all_logits)
    #     labels = torch.cat(all_labels)
    #     subgroup = torch.cat(all_groups)      # [N, K]
    #     cls_all = torch.cat(all_cls)          # [N, H]

    #     preds = logits.argmax(-1)

    #     # basic metrics
    #     acc = accuracy_score(labels.cpu(), preds.cpu())
    #     p = precision_score(labels.cpu(), preds.cpu())
    #     r = recall_score(labels.cpu(), preds.cpu())
    #     f1 = f1_score(labels.cpu(), preds.cpu())

    #     # FPR
    #     fp = ((preds == 1) & (labels == 0)).sum().item()
    #     tn = ((preds == 0) & (labels == 0)).sum().item()
    #     fpr = fp / (fp + tn + 1e-8)

    #     # fairness metric: weighted mean projections
    #     v = trainer.v_tox.to(device)
    #     v = v / (v.norm() + 1e-8)

    #     proj = (cls_all * v).sum(-1)        # [N]

    #     K = subgroup.shape[1]
    #     fairness_scores = {}

    #     for k in range(K):
    #         w_k = subgroup[:, k]
    #         numer = (w_k * proj).sum()
    #         denom = w_k.sum() + 1e-8
    #         fairness_scores[f"subgroup_{k}_bias"] = (numer / denom).item()

    #     # combine results
    #     result = {
    #         "accuracy": acc,
    #         "precision": p,
    #         "recall": r,
    #         "f1": f1,
    #         "FPR_overall": fpr,
    #     }
    #     result.update(fairness_scores)
    #     return result



    def train(self, ds):

        train_dataset, valid_dataset = self.prepare_dataset(ds, batch_size=8)
        print("====== DATASET PREPARED ========")

        training_args = TrainingArguments(
            output_dir="fair-mp-bert",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            fp16=True,
            # dataloader_num_workers=2,
            gradient_checkpointing=True,
            # lambda_orth=0.3,
            remove_unused_columns=False

        )



        def data_collator(batch):
            # pad input_ids and attention_mask
            # for b in batch:
            #     print(b.keys())
            input_ids = [b["input_ids"] for b in batch]
            attention = [b["attention_mask"] for b in batch]
            
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention = torch.nn.utils.rnn.pad_sequence(attention, batch_first=True, padding_value=0)

            labels = torch.tensor([b["label"] for b in batch])
            subgroup = torch.stack([b["weights"] for b in batch])

            return {
                "input_ids": input_ids,
                "attention_mask": attention,
                "labels": labels,
                "weights": subgroup
            }

        callback = ToxicDirectionCallback()

        trainer = FairnessTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator= data_collator,
            # callbacks=[ToxicDirectionCallback()]
            callbacks = [callback]
        )

        callback.trainer = trainer

        trainer.lambda_orth = 0.3 
        trainer.v_tox = torch.zeros(self.hidden_size)


        trainer.train()


        save_path= save_path + "/best"
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved locally at: {save_path}")

        repo_name = "eunie922/" + save_path
        create_repo(repo_name)

        if repo_name is None:
            raise ValueError("`repo_name` must be provided for HuggingFace Hub upload.")

        print(f"Pushing model to HuggingFace Hub repo: {repo_name} ...")

        self.model.push_to_hub(repo_name)
        self.tokenizer.push_to_hub(repo_name)

        print(f"Successfully uploaded to https://huggingface.co/{repo_name}")

        
        print("=== Final Evaluation ===")

        eval_results = self.evaluate(trainer, valid_dataset)
        for k, v in eval_results.items():
            print(f"{k}: {v:.5f}")





if __name__ == "__main__":
    
    # df = pd.read_csv(r"datasets/JIGSAW/train.csv")
    # print(df.head())

    df = load_from_disk("preprocessed_train")

    race_ds = toxic_preprocess(df, "race")
    trainer = FairToxic("bert-base-uncased", race_ds)
    trainer.train(race_ds)




