import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm
from tqdm import trange
import numpy as np
import optuna
import os

# ======== Constants ========
MAX_LEN = 30
BATCH_SIZE = 32
OUTPUT_DIM = 6

# ======== Preprocessing ========
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def tokenize(text):
    return text.split()

def encode_tokens(tokens, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

def pad_sequence(seq, max_len, pad_idx):
    return seq + [pad_idx] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]

def load_and_prepare_data(path, vocab=None):
    df = pd.read_csv(path)
    df["clean_statement"] = df["statement"].apply(clean_text)
    df["tokens"] = df["clean_statement"].apply(tokenize)

    if vocab is None:
        all_tokens = [token for tokens in df["tokens"] for token in tokens]
        token_counts = Counter(all_tokens)
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for i, (token, _) in enumerate(token_counts.items(), start=2):
            vocab[token] = i

    df["input_ids"] = df["tokens"].apply(lambda x: encode_tokens(x, vocab))
    df["padded_ids"] = df["input_ids"].apply(lambda x: pad_sequence(x, MAX_LEN, vocab["<PAD>"]))
    return df, vocab

# ======== Dataset ========
class Liar2Dataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# ======== Model ========
class FakeNewsClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

# ======== Objective Function for Optuna ========
def objective(trial):
    embed_dim = trial.suggest_categorical("embedding_dim", [50, 100, 128])
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    epochs = 3  # short for tuning

    train_df, vocab = load_and_prepare_data("../../data/processed/liar2_train.csv")
    dataset = Liar2Dataset(train_df["padded_ids"].tolist(), train_df["label"].tolist())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FakeNewsClassifier(len(vocab), embed_dim, hidden_dim, OUTPUT_DIM, vocab["<PAD>"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    best_acc = 0.0

    for _ in range(epochs):
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)

        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            save_dir = r"C:\Users\Zayed\Documents\GitHub\FactuAI\models"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

    return best_acc

def main():
    study = optuna.create_study(direction="maximize")
    n_trials = 10

    for _ in trange(n_trials, desc="Optimizing"):
        study.optimize(objective, n_trials=1, catch=(Exception,))  # 1 trial at a time

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value:.4f}")
    for k, v in trial.params.items():
        print(f"  {k}: {v}")
        
if __name__ == "__main__":
    main()