from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from datasets import Dataset


from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSequenceClassification
)



# EMBEDDING

@torch.no_grad()
def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_embeddings(model_path, texts, method="mean", max_length=256, l2norm=True):
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
