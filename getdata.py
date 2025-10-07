from datasets import load_dataset
import os
os.makedirs("data/wikitext-103", exist_ok=True)

ds = load_dataset("wikitext", "wikitext-103-v1")
ds["train"].to_pandas().to_csv("data/wikitext-103/train.txt", sep="\n", header=False, index=False)
ds["validation"].to_pandas().to_csv("data/wikitext-103/valid.txt", sep="\n", header=False, index=False)
ds["test"].to_pandas().to_csv("data/wikitext-103/test.txt", sep="\n", header=False, index=False)