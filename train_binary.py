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
    df["label"] = (df["target"] > 0.5).astype(int)
    df = df.rename(columns={"comment_text": "text", "target": "toxicity"})
    return Dataset.from_pandas(df)


class binaryClassifier:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(self.device)

    def prepare_train(self, ds, test_size=0.2, seed=42):
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

        save_path = "./bert-bin-classifier/best"
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"\n✅ Model saved to: {save_path}")

        return save_path



if __name__ == "__main__":

    df = pd.read_csv(r"datasets/JIGSAW/train.csv")
    ds = preprocess(df)

    model_name = "bert-base-uncased"
    bc = binaryClassifier(model_name)
    model_path = bc.train(ds)


