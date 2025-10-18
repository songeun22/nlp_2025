import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from evaluate import load
from huggingface_hub import login




# preprocessed data with binary label 


def preprocess(df):
    df = df[["comment_text", "target"]].copy()
    df = df.dropna(subset=["comment_text", "target"])
    df["label"] = (df["target"] > 0).astype(int)
    df = df.rename(columns={"comment_text": "text", "target": "toxicity"})
    return Dataset.from_pandas(df)


class binaryClassifier:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(self.device)

    def prepare_train(self, ds, test_size=0.2, seed=42):
        """
        단일 Dataset을 받아 train/test로 나누고 토크나이징까지 수행.
        """

        ds_split = ds.train_test_split(test_size=test_size, seed=seed)
        train_ds = ds_split["train"]
        test_ds = ds_split["test"]

        def tokenize(batch):
            return self.tokenizer(list(batch["text"]), padding="max_length", truncation=True, max_length=128)

        train_ds = train_ds.map(tokenize, batched=True)
        test_ds = test_ds.map(tokenize, batched=True)

        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        return train_ds, test_ds
    
    def train(self, ds):
        train_ds, test_ds = self.prepare_train(ds)

        accuracy = load("accuracy")
        precision = load("precision")
        recall = load("recall")
        f1 = load("f1")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
                "precision": precision.compute(predictions=preds, references=labels)["precision"],
                "recall": recall.compute(predictions=preds, references=labels)["recall"],
                "f1": f1.compute(predictions=preds, references=labels)["f1"],
            }

        training_args = TrainingArguments(
            output_dir="./bert-bin-classifier",
            eval_strategy="steps",
            eval_steps=10000,
            save_strategy="steps",
            save_steps=10000,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=8,
            num_train_epochs=1,
            weight_decay=0.01,
            load_best_model_at_end=True,
            logging_dir="./logs",
            logging_steps=50,
            fp16 = True, 
            dataloader_num_workers = 2
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()
        print("\n📊 Evaluation Metrics:", metrics)

        # ✅ 모델 저장
        save_path = "./bert-bin-classifier/best"
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"\n✅ Model saved to: {save_path}")

        return save_path


# EMBEDDING

@torch.no_grad()
def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_embeddings(model_path, texts, method="mean", max_length=128, l2norm=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()

    enc = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    out = model(**enc, return_dict=True)

    if method == "mean":
        sent = mean_pooling(out.last_hidden_state, enc["attention_mask"])
    elif method == "cls":
        sent = out.last_hidden_state[:, 0, :]  # [CLS]
    else:
        raise ValueError("method must be 'mean' or 'cls'")

    if l2norm:
        sent = torch.nn.functional.normalize(sent, p=2, dim=-1)

    return sent.cpu().numpy()


def visualize_embeddings(emb, labels, title, type):
    pca = PCA(n_components=50, random_state=42)
    X50 = pca.fit_transform(emb)
    tsne = TSNE(n_components=2, metric="cosine", perplexity=5, init="pca", random_state=42)
    Z = tsne.fit_transform(X50)

    plt.figure(figsize=(6, 5))

    if type == 'binary': 
        mask = np.array(labels) == 1
        plt.scatter(Z[mask, 0], Z[mask, 1], color="red", label="Label 1", marker="+")
        plt.scatter(Z[~mask, 0], Z[~mask, 1], color="blue", label="Label 0", marker="+")
        plt.legend()
    

    elif type == 'conti': 
        plt.scatter(Z[:, 0], Z[:, 1], c = labels, cmap = "RdPu", alpha = 0.8, s = 15)
        plt.colorbar(label = "toxicity")

    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 4️⃣ 실행 예시
# ============================================================

if __name__ == "__main__":
    # ✅ CSV 로드
    df = pd.read_csv(r"datasets/JIGSAW/train.csv")
    ds = preprocess(df)

    # ✅ 학습 및 저장
    model_name = "bert-base-uncased"
    bc = binaryClassifier(model_name)
    model_path = bc.train(ds)

    # ✅ 임베딩 추출
    n_sample = 500
    sample = ds.shuffle(seed=42).select(range(n_sample))
    texts = list(sample["text"])
    labels = np.array(sample["label"])

    # mean pooling
    emb_mean = encode_embeddings(model_path, texts, method="mean")
    visualize_embeddings(emb_mean, labels, "t-SNE of Mean-Pooled Sentence Embeddings", "binary")

    # CLS embedding
    emb_cls = encode_embeddings(model_path, texts, method="cls")
    visualize_embeddings(emb_cls, labels, "t-SNE of [CLS] Token Embeddings", "binary")

    labels = np.array(sample["toxicity"])
    # mean pooling
    emb_mean = encode_embeddings(model_path, texts, method="mean")
    visualize_embeddings(emb_mean, labels, "t-SNE of Mean-Pooled Sentence Embeddings", "conti")

    # CLS embedding
    emb_cls = encode_embeddings(model_path, texts, method="cls")
    visualize_embeddings(emb_cls, labels, "t-SNE of [CLS] Token Embeddings", "conti")
