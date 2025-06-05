from datasets import load_dataset
import pandas as pd

print("📦 Loading LIAR2 dataset...")
ds = load_dataset("chengxuphd/liar2")

print("📄 Converting to DataFrames...")
train_df = pd.DataFrame(ds["train"])
val_df = pd.DataFrame(ds["validation"])
test_df = pd.DataFrame(ds["test"])

print("💾 Saving to CSV...")
train_df.to_csv("../data/processed/liar2_train.csv", index=False)
val_df.to_csv("../data/processed/liar2_val.csv", index=False)
test_df.to_csv("../data/processed/liar2_test.csv", index=False)

print("✅ All done. CSV files saved in /data/processed/")
