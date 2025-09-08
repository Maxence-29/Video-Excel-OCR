import torch
from torch.utils.data import Dataset
import json

class OCRCorrectionDataset(Dataset):
    def __init__(self, filepath, vocab_path="vocab.json", max_len=16):
        with open(filepath, encoding="utf-8") as f:
            self.pairs = [line.strip().split('\t') for line in f if '\t' in line]

        with open(vocab_path, encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.id2char = {v: k for k, v in self.vocab.items()}
        self.pad_id = self.vocab["<PAD>"]
        self.unk_id = self.vocab["<UNK>"]
        self.max_len = max_len

    def encode(self, text):
        ids = [self.vocab.get(ch, self.unk_id) for ch in text]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids += [self.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids)

    def __getitem__(self, idx):
        noisy, clean = self.pairs[idx]
        return self.encode(noisy), self.encode(clean)

    def __len__(self):
        return len(self.pairs)