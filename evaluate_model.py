import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# -------------------------------
# Load tokenized test dataset
# -------------------------------

print("[INFO] Loading test dataset...")
from torch.serialization import add_safe_globals
from transformers.tokenization_utils_base import BatchEncoding
add_safe_globals({"transformers.tokenization_utils_base.BatchEncoding": BatchEncoding})
test_encodings, test_labels = torch.load("../data/test_dataset.pt", weights_only=False)

def make_dataset(encodings, labels):
    return TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        labels
    )

test_dataset = make_dataset(test_encodings, test_labels)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -------------------------------
# Load trained model + tokenizer
# -------------------------------
print("[INFO] Loading trained model...")
model_path = "../models/securox_distilbert"

model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
model.to(device)
model.eval()

# -------------------------------
# Evaluate
# -------------------------------
print("[INFO] Evaluating on test set...")
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# -------------------------------
# Metrics
# -------------------------------
acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="weighted"
)

print("\n===== Test Results =====")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# -------------------------------
# Confusion matrix
# -------------------------------
cm = confusion_matrix(all_labels, all_preds)
labels = ["legit", "spam", "phishing"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - SECUROX DistilBERT")

save_path = "../models/confusion_matrix.png"
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"[INFO] Confusion matrix saved to {save_path}")
