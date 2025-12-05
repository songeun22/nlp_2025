# nlp_2025
SNU 2025 Fall / Introduction to NLP / Project codes

### **1. Baseline: BERT - evaluation**

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
