import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import OCRCorrectionDataset
from model import CharCorrector
import json
import os

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab
    with open("vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    pad_id = vocab["<PAD>"]

    # Datasets & loaders
    train_ds = OCRCorrectionDataset("data/train.txt")
    val_ds = OCRCorrectionDataset("data/val.txt")
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)

    # Model & optimizer
    model = CharCorrector(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # Early stopping parameters
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(20):
        model.train()
        total_loss = 0
        for noisy, clean in train_dl:
            noisy, clean = noisy.to(device), clean.to(device)
            logits = model(noisy)  # [B, T, V]
            loss = criterion(logits.view(-1, vocab_size), clean.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dl)

        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for noisy, clean in val_dl:
                noisy, clean = noisy.to(device), clean.to(device)
                pred = model(noisy).argmax(dim=-1)
                mask = (clean != pad_id)
                correct += (pred == clean)[mask].sum().item()
                total += mask.sum().item()
        acc = correct / total
        print(f"[Epoch {epoch+1}] loss={avg_loss:.4f} val_acc={acc:.4f}")

        if acc > best_val_acc:
            best_val_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/digit_corrector.pt")
            print(f"[✓] New best model saved!")
        else:
            patience_counter += 1
            print(f"[!] No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("[x] Early stopping triggered.")
                break

if __name__ == "__main__":
    train()