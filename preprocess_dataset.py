import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
import re

# -------------------------------
# Load dataset
# -------------------------------
print("[INFO] Loading dataset...")
df = pd.read_csv("../data/emails_merged_balanced.csv")  # update to correct file
print(df.columns)
print("[INFO] Dataset shape:", df.shape)
print(df['label'].value_counts())



# -------------------------------
# Map labels to integers
# -------------------------------
label_map = {"legit": 0, "spam": 1, "phishing": 2}
df["label"] = df["label"].map(label_map).astype(int)
print("[DEBUG] Label dtype:", df["label"].dtype)
print("[DEBUG] Unique labels:", df["label"].unique())
# -------------------------------
# Simple cleaning function
# -------------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)        # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)        # remove emails
    text = re.sub(r"[^a-z\s]", " ", text)       # keep only letters/spaces
    text = re.sub(r"\s+", " ", text).strip()    # remove extra spaces
    return text

# -------------------------------
# Apply cleaning
# -------------------------------
print("[INFO] Cleaning text...")
df["text_clean"] = df["text"].apply(clean_text)

# -------------------------------
# Train/Validation/Test Split
# -------------------------------
df_train, temp_texts, train_labels, temp_labels = train_test_split(
    df[["text_clean", "label"]], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

df_val, df_test, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"[INFO] Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

# -------------------------------
# Tokenization
# -------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(df_train["text_clean"].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
val_encodings = tokenizer(df_val["text_clean"].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
test_encodings = tokenizer(df_test["text_clean"].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")

train_labels = torch.tensor(df_train["label"].values)
val_labels = torch.tensor(df_val["label"].values)
test_labels = torch.tensor(df_test["label"].values)

# -------------------------------
# Save tokenized datasets
# -------------------------------
torch.save((train_encodings, train_labels), "../data/train_dataset.pt")
torch.save((val_encodings, val_labels), "../data/val_dataset.pt")
torch.save((test_encodings, test_labels), "../data/test_dataset.pt")

print("[INFO] Tokenized datasets saved in data/ folder.")
print("[INFO] Example tokenized input IDs:", train_encodings['input_ids'][0])