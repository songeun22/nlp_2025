from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch


from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix




group_colnames = {
    "gender": ['female', 'male', 'transgender', 'other_gender'], 
    "race" : ["black", "white" ,"asian", "latino", "other_race_or_ethnicity"], 
    "religion" : ["christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion"]
}







def print_eval(true_label, pred_label): 

    cf =pd.crosstab(
        true_label, pred_label, rownames = ["Toxic"], colnames = ["Pred"]
        )
    
    accuracy = accuracy_score(true_label, pred_label)
    precision = precision_score(true_label, pred_label)
    recall = recall_score(true_label, pred_label)
    f1 = f1_score(true_label, pred_label)

    print("=== Confusion Matrix ===")
    print(cf)

    print("\n=== Basic Metrics ===")
    print({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })




def toxic_preprocess(df, group):
    """
    Preprocess dataframe with given group name

    Input:
        df: original pandas dataframe
        group: group name ["gender", "race", "religion"]
    """
    
    df = df.copy()
    df = df.dropna(subset=["comment_text", "target"])
    df["label"] = np.array(df["target"] >= 0.5, dtype = int)
    df = df.rename(columns={"comment_text": "text", "target": "toxicity"})

    group_colnames = {
        "gender": ['female', 'male', 'transgender', 'other_gender'], 
        "race" : ["black", "white" ,"asian", "latino", "other_race_or_ethnicity"], 
        "religion" : ["christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion"]
    }

    if group is not None: 
        selected_colnames = group_colnames[group]
        df["subgroup_weights"] = df[selected_colnames].values.tolist()
        df = df.loc[np.sum(df["subgroup_weights"]) > 0] 
        df = df[["text", "toxicity", "label"] + selected_colnames]

    return Dataset.from_pandas(df)



def toxic_preprocess(df, group):
    """
    Preprocess dataframe with given group name, converted to Hugging Face Dataset.

    Input:
        df: original pandas dataframe (for group = None) 
        df: original dataset (in local disk)
        group: group name ["gender", "race", "religion"]
    """

    if group is None: 

        dataset = Dataset.from_pandas(df)

        dataset = dataset.filter(lambda x: x['comment_text'] is not None and x['target'] is not None)
        dataset = dataset.map(lambda x: {"label": int(x["target"] >= 0.5)})
        
        dataset = dataset.rename_column("comment_text", "text")
        dataset = dataset.rename_column("target", "toxicity")

    group_colnames = {
        "gender": ['female', 'male', 'transgender', 'other_gender'], 
        "race" : ["black", "white" ,"asian", "latino", "other_race_or_ethnicity"], 
        "religion" : ["christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion"]
    }

    if group is not None:

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
        
        group_ds = df.map(add_weight_array)

        group_ds = group_ds.filter(lambda x: sum(x["weights"]) > 0)
        dataset = group_ds.select_columns(["text", "toxicity", "label", "weights"] + selected_colnames)

    return dataset





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
    tok = AutoTokenizer.from_pretrained(model_ckpt)

    def tokenize(ex):
        out = tok(
            ex["text"],
            padding="max_length",
            truncation=True,
            max_length=256
        )
        return out
    
    tokenized_ds = check_ds.map(tokenize, batched=True)

    bsz = 100
    probs = []
    for batch in tokenized_ds.with_format("torch").iter(batch_size=bsz):
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




def eval_baseline(eval_ds, group_name):

    cols = group_colnames[group_name]
    all_cols = cols + ["text", "label", "toxicity", "preds", "scores"]
    group_ds = eval_ds.select_columns(all_cols)

    def add_weight_array(elmt): 
        elmt["weights"] = np.array([elmt[c] for c in cols])
        return elmt 

    group_ds = group_ds.map(add_weight_array)
    
    group_ds = group_ds.filter(lambda e: np.sum(e["weights"]) > 0)

    for col in group_colnames[group_name]:
        sub_ds = group_ds.filter(lambda e: e[col] > 0)

        cf = confusion_matrix(sub_ds["label"], sub_ds["preds"])
        accuracy = accuracy_score(sub_ds["label"], sub_ds["preds"])
        precision = precision_score(sub_ds["label"], sub_ds["preds"])
        recall = recall_score(sub_ds["label"], sub_ds["preds"])
        f1 = f1_score(sub_ds["label"], sub_ds["preds"])

        print(f"=== {col} RESULTS ===\n")

        print("=== Confusion Matrix ===")
        print(cf)

        print("\n=== Basic Metrics ===")
        print({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
    
    return group_ds



