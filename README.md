# Unintended Bias in Toxicity Classification 
### : analyzing and mitigating representational bias in toxicity models

SNU 2025 Fall / Introduction to NLP / Project codes




### Dataset

https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview


Here's the pipeline of overall evaluation. \
(check the fair.tools.py for details) 

## **1. Baseline: BERT - evaluation**

```
# === Preprocess the original dataset & save ! ===

# df = pd.read_csv(r"datasets/JIGSAW/train.csv")
# ds = Dataset.from_pandas(df)
# ds = toxic_preprocess(ds, None)
# ds.save_to_disk("preprocessed_train")


# === Split train/test dataset in baseline classifier === 

ds = load_from_disk("datasets/preprocessed_all")

processed = ds.train_test_split(test_size=0.2, seed=42)
train_ds = processed["train"]
test_ds = processed["test"]  # eval set
```


```

# === Infer predictions and add "preds" column === 

model_ckpt = "eunie922/bert-toxic-jigsaw"  # baseline classifier
# test_result = get_pred(test_ds, model_ckpt)
# test_result.save_to_disk("baseline_eval")  # save 


# === print eval metrics for each group  === 

test_result = load_from_disk("datasets/baseline_eval")  # load inferred baseline dataset
print("\n\n 1. GENDER \n\n")
gender_baseline, gender_dict = subgroup_eval(test_result, "gender", True, True)
get_gap(gender_dict)   # get FPR, TPR gaps 
print("\n\n 2. RELIGION \n\n")relig_baseline, relig_dict = subgroup_eval(test_result, "religion", True, True)
get_gap(relig_dict)
print("\n\n 3. RACE \n\n")
race_baseline, race_dict = subgroup_eval(test_result, "race", True, True)
get_gap(race_dict)
```

For embedding analysis, use ```id_emb_dist``` function. 

```
relig_dist = id_emb_dist(relig_baseline, "religion", model_ckpt, True)
print(relig_dist)
```


---

## **2. FAIR Train evaluation**

```
# df = load_from_disk("datasets/preprocessed_all")
relig_ds, no_id_ds = toxic_preprocess(df, "religion", False)
eval_ds = relig_ds.train_test_split(test_size=0.2, seed=2025)
eval_ds = concatenate_datasets([eval_ds["test"], no_id_ds])

model_ckpt = "fair_path/relig_001/best"
eval_pred = get_pred(eval_ds, model_ckpt, False)
relig_res, relig_metric = subgroup_eval(eval_pred, "religion", True, True)
```



