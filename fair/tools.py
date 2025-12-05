from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModel
)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix




group_colnames = {
    "gender": ['female', 'male', 'transgender', 'other_gender'], 
    "race" : ["black", "white" ,"asian", "latino", "other_race_or_ethnicity"], 
    "religion" : ["christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion"]
}





def print_eval(true_label, pred_label): 

    cf = confusion_matrix(true_label, pred_label)
    tn, fp, fn, tp = cf.ravel()
    
    accuracy = accuracy_score(true_label, pred_label)
    precision = precision_score(true_label, pred_label)
    recall = recall_score(true_label, pred_label)
    f1 = f1_score(true_label, pred_label)
    fpr = fp / (fp + tn + 1e-8)
    tpr = tp / (tp + fn + 1e-8)

    print("=== Confusion Matrix ===")
    print(cf)

    print("\n=== Basic Metrics ===")
    print(f"accuracy: {accuracy:.4f}\n" + 
          f"precision: {precision:.4f}\n" +
          f"recall: {recall:.4f}\n" +
          f"f1: {f1:.4f}\n" +
          f"fpr: {fpr:.4f}\n" +
          f"tpr : {tpr:.4f}\n")




def toxic_preprocess(dataset, group, eval):
    """
    Preprocess dataframe with given group name, converted to Hugging Face Dataset.

    Input:
        dataset: Dataset
        group: group name [None, "gender", "race", "religion"] - if None, it's for baseline classifier 
        eval: True if you need to include "preds" column
    """

    group_colnames = {
        "gender": ['female', 'male', 'transgender', 'other_gender'], 
        "race" : ["black", "white" ,"asian", "latino", "other_race_or_ethnicity"], 
        "religion" : ["christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion"]
    }

    if group is None: 
        dataset = dataset.filter(lambda x: x['comment_text'] is not None and x['target'] is not None)
        dataset = dataset.map(lambda x: {"label": int(x["target"] >= 0.5)})
        
        dataset = dataset.rename_column("comment_text", "text")
        dataset = dataset.rename_column("target", "toxicity")

        return dataset

    else:
        selected_colnames = group_colnames[group]

        def add_weight_array(elmt): 
            arr = []
            for c in selected_colnames:
                v = elmt[c]
                if v is None:
                    v = 0
                elif isinstance(v, float) and np.isnan(v):
                    v = 0
                arr.append(v)
            elmt["weights"] = np.array(arr, dtype=float)
            return elmt
        
        group_ds = dataset.map(add_weight_array)

        id_ds = group_ds.filter(lambda x: sum(x["weights"]) > 0)
        no_id_ds = group_ds.filter(lambda x: sum(x["weights"]) == 0)
        
        if eval: # after get_pred
            selected_colnames += ["text", "toxicity", "label", "weights", "preds"]
        else:  # before get_pred
            selected_colnames += ["text", "toxicity", "label", "weights"]

        id_ds = id_ds.select_columns(selected_colnames)
        no_id_ds = no_id_ds.select_columns(selected_colnames)
    
        return id_ds, no_id_ds




def get_pred(check_ds, model_ckpt, return_metric = True):
    """
    push predicted labels to input dataset 

    Input:
        check_ds: input dataset with target texts("text") and true labels("label")-> Dataset
        model_ckpt: model checkpoint or path -> str
        return_metric: returns confusion matrix -> Bool
        
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    machine = AutoModelForSequenceClassification.from_pretrained(model_ckpt).to(device).eval()
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(ex):
        out = tok(
            ex["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors = "pt"
        )

        return {
            "input_ids" : out["input_ids"], 
            "attention_mask" : out["attention_mask"]
        }
    
    tokenized_ds = check_ds.map(
        tokenize,
        batched=True,
        remove_columns=check_ds.column_names  
    )

    bsz = 50
    probs = []

    with torch.no_grad(): 
        for batch in tokenized_ds.iter(batch_size=bsz):
            batch = {k: torch.tensor(v).to(device) for k, v in batch.items()}
            out = machine(**batch)
            logits = out.logits
            batch_probs = torch.softmax(logits, axis = -1).cpu()
            probs.extend(batch_probs[:, 0].tolist())

    pred_labels = (np.array(probs) < 0.5).astype(int)
    check_ds = check_ds.add_column("scores", 1 - np.array(probs))
    check_ds = check_ds.add_column("preds", pred_labels)

    if return_metric:
        print_eval(check_ds["label"], pred_labels)

    return check_ds






def subgroup_eval(eval_ds, group_name, preprocess, verbose):
    """
    prints evaluation metrics for each subgroups

    eval_ds : input Dataset with "preds"
    group_name : "gender", "religion", "race" 
    preproces : True if input eval_ds needs to be preprocessed
    verbose : print the mterics 
    """
    
    if preprocess: 
        group_ds, no_id_ds = toxic_preprocess(eval_ds, group_name, True)
    else:
        group_ds = eval_ds

    if verbose: 
        print_eval(eval_ds["label"], eval_ds["preds"])

    metric_dict = {}
    all_cols = group_colnames[group_name] + ["no_id"]

    for i, col in enumerate(all_cols):

        if i == len(all_cols) - 1: 
            sub_ds = no_id_ds
            # sub_ds = group_ds.filter(lambda e: torch.sum(torch.tensor(e["weights"])) == 0)
        else:
            sub_ds = group_ds.filter(lambda e: float(e["weights"][i]) >= 0.5)

        cf = confusion_matrix(sub_ds["label"], sub_ds["preds"])
        tn, fp, fn, tp = cf.ravel()
        accuracy = accuracy_score(sub_ds["label"], sub_ds["preds"])
        precision = precision_score(sub_ds["label"], sub_ds["preds"])
        recall = recall_score(sub_ds["label"], sub_ds["preds"])
        f1 = f1_score(sub_ds["label"], sub_ds["preds"])
        fpr = fp / (fp + tn + 1e-8)
        tpr = tp / (tp + fn + 1e-8)

        metric_dict[col] = {
                "accuracy": accuracy,
                "precision":  precision,
                "recall":   recall,
                "f1":  f1, 
                "FPR": fpr, 
                "TPR" : tpr
            }
        
    
        if verbose:
            print(f"\n\n=== {col} RESULTS ===\n")

            print("=== Confusion Matrix ===")
            print(cf)

            print("\n=== Basic Metrics ===")
            print({
                "accuracy":  f"{accuracy:.4f}",
                "precision": f"{precision:.4f}",
                "recall":    f"{recall:.4f}",
                "f1":        f"{f1:.4f}",
                "FPR":       f"{fpr:.4f}", 
                "TPR": f"{tpr:.4f}"
            })
    
    return group_ds, metric_dict



def get_gap(metric_dict):
    
    FPR_list = [] 
    TPR_list = []
    for g, m in metric_dict.items():
        FPR_list.append(m["FPR"])
        TPR_list.append(m["TPR"])

    print(f"FPR GAP: {np.max(FPR_list) - np.min(FPR_list):.4f}")
    print(f"TPR GAP: {np.max(TPR_list) - np.min(TPR_list):.4f}")





# EMBEDDING

@torch.no_grad()
def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts
    

@torch.no_grad()
def encode_embeddings(texts, model, max_length=256, return_probs=True, batch_size=50):
    all_embeds = {"mean": [], "cls": []}
    all_probs = []
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        enc = tok(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        
        out = model(**enc, return_dict=True)

        for method in ["mean", "cls"]:
            if method == "mean":
                sent = mean_pooling(out.last_hidden_state, enc["attention_mask"])
            elif method == "cls":
                sent = out.last_hidden_state[:, 0, :]

            # sent = torch.nn.functional.normalize(sent, p=2, dim=-1)
            all_embeds[method].append(sent.cpu())

        if return_probs:
            out2 = model(**enc)
            probs = torch.softmax(out2.logits, dim=-1)
            all_probs.append(probs.cpu())

        del enc, out
        if return_probs:
            del out2, probs
    torch.cuda.empty_cache()

    for method in ["mean", "cls"]:
        all_embeds[method] = torch.cat(all_embeds[method], dim=0).numpy()

    if return_probs:
        all_probs = torch.cat(all_probs, dim=0).numpy()
        return all_embeds, all_probs
    else:
        return all_embeds




def visualize_embeddings(emb, labels, title, legend_title, type):
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
    plt.legend(title = legend_title)
    plt.tight_layout()
    plt.show()




# === MEMO ===
# 1. toxicity direction: 
# use true label vs. preds or weight by probabilities? or weight by toxicity (given in dataset) 
# 2. group embeddings: 
# use discrete indicator (cut off by 0.5) ? or weights ? 


def toxic_embeddings(model_ckpt, eval_ds, group_name, sample, method = "cls"):
    """
    Computes toxicity embedding direction (v_tox) and weighted embedding vectors (wm_emb)
    
    model_ckpt : model checkpoint 
    eval_ds : dataset with "label", "weights", "text"
    group_name : "gender", "religion", "race"
    sample : True if sample 2000 texts 
    method : embedding methods -> "cls" or "mean" 
    """

    subgroup_names = group_colnames[group_name]
    
    if sample: 
        n_sample = 2000
        eval_ds = eval_ds.shuffle(seed=42).select(range(n_sample))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_ckpt).to(device).eval()

    texts = list(eval_ds["text"])
    labels = np.array(eval_ds["label"])
    # labels = np.array(sample["preds"])     # [N]


    emb = encode_embeddings(texts, model, return_probs = False)
    emb = emb[method] # [N, 768]

    mu_tox = emb[labels == 1].mean(axis=0)
    mu_non = emb[labels == 0].mean(axis=0)
    v_tox = mu_tox - mu_non

    if group_name != "no_id": 
    
        weights = np.array(eval_ds["weights"])  # [N, K]

        weighted_sum = emb.T @ weights    # [768, K]
        wm_emb = weighted_sum / weights.sum(axis=0)  # weighted mean of embedding vectors for each identities

        return v_tox, wm_emb
    
    else: 
        
        return v_tox, emb


        

def id_emb_dist(ds, group_name, model_ckpt, plot = True):
    """
    Docstring for id_emb_dist
    
    :param ds: Dataset; all rows with identity mentioned! (toxic process with group name)
    :param group_name: gender, race, religion
    :param model_ckpt: trained model checkpoint
    """

    v_tox, emb_mat = toxic_embeddings(model_ckpt, ds, group_name, sample = False) # gender_baseline, relig_baseline 
    subgroup_names = group_colnames[group_name]

    # norm of v_tox : how far away are toxic/non-toxic clusters located? 
    print(f"{np.linalg.norm(v_tox):.4f}")

    # subgroup distance matrix 
    K = emb_mat.shape[1]
    dist_mat = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            diff = emb_mat[:, i] - emb_mat[:, j]
            dist_mat[i, j] = np.linalg.norm(diff)


    plt.figure(figsize=(6, 5))

    mask = np.triu(np.ones_like(dist_mat, dtype=bool), k=1) 

    dist_show = np.ma.array(dist_mat, mask=mask)

    # --- heatmap ---
    cmap = plt.cm.PuBuGn
    cmap.set_bad(color="white") 

    for i in range(K):
        for j in range(K):
            if not mask[i, j]: 
                plt.text(
                    j, i,
                    f"{dist_mat[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if dist_mat[i, j] > dist_mat.max()/2 else "black",
                    fontsize=10
                )

    if plot: 
        plt.imshow(dist_show, cmap=cmap)

        plt.title("Pairwise Distance Matrix")
        plt.colorbar(label="L2 distance")

        plt.xlabel("Identity index")
        plt.ylabel("Identity index")
        plt.tight_layout()

        plt.xticks(ticks=np.arange(K), labels=subgroup_names, rotation=45, ha="right")
        plt.yticks(ticks=np.arange(K), labels=subgroup_names)

        plt.show()

    return dist_mat
