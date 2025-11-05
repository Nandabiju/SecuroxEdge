import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from tqdm import tqdm
import os

# -------------------------------
# Load tokenized datasets
# -------------------------------
print("[INFO] Loading datasets...")
train_encodings, train_labels = torch.load("../data/train_dataset.pt", weights_only=False)
val_encodings, val_labels = torch.load("../data/val_dataset.pt", weights_only=False)
test_encodings, test_labels = torch.load("../data/test_dataset.pt", weights_only=False)

def make_dataset(encodings, labels):
    return TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        labels
    )

train_dataset = make_dataset(train_encodings, train_labels)
val_dataset = make_dataset(val_encodings, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
print("[INFO] Datasets ready.")

# -------------------------------
# Define model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3  # legit, spam, phishing
)
model.to(device)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# -------------------------------
# Optimizer
# -------------------------------
optimizer = AdamW(model.parameters(), lr=5e-5)

# -------------------------------
# Training loop with early stopping
# -------------------------------
epochs = 4
best_val_acc = 0
patience = 1  # stop if no improvement after 1 epoch
patience_counter = 0

save_path = "../models/securox_distilbert"
os.makedirs(save_path, exist_ok=True)

for epoch in range(epochs):
    print(f"\n===== Epoch {epoch+1}/{epochs} =====")

    # Training
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Train loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Check for improvement
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # Save best model + tokenizer
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"[INFO] New best model saved with val acc {val_acc:.4f}")
    else:
        patience_counter += 1
        print(f"[INFO] No improvement. Patience counter: {patience_counter}")

    if patience_counter >= patience:
        print("[INFO] Early stopping triggered.")
        break

print(f"\n[INFO] Training complete. Best validation accuracy: {best_val_acc:.4f}")
print(f"[INFO] Best model saved to {save_path}")
