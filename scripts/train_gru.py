import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from src.model import GRURegressor
from src.preprocess import add_rul, load_data, make_sequences, scale_features

SEQ_LEN = 30
RUL_CAP = 130
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
SEED = 42

FEATURE_COLS = ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class RULDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    set_seed(SEED)

    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "train_FD001.txt"
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    print("Using device:", get_device())

    df = load_data(str(data_path))
    df = add_rul(df)
    df, scaler = scale_features(df, FEATURE_COLS)

    X, y = make_sequences(df, FEATURE_COLS, seq_len=SEQ_LEN, rul_cap=RUL_CAP)

    dataset = RULDataset(X, y)

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    device = get_device()
    model = GRURegressor(input_size=len(FEATURE_COLS)).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def evaluate(loader):
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                total_loss += loss.item() * xb.size(0)

        mse = total_loss / len(loader.dataset)
        rmse = mse**0.5
        return mse, rmse

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_mse = running_loss / len(train_loader.dataset)
        val_mse, val_rmse = evaluate(val_loader)

        print(
            f"Epoch {epoch:02d} | "
            f"Train MSE: {train_mse:.4f} | "
            f"Val MSE: {val_mse:.4f} | "
            f"Val RMSE: {val_rmse:.4f}"
        )

    model_path = models_dir / "gru_rul_fd001.pt"
    scaler_path = models_dir / "gru_scaler_fd001.joblib"
    meta_path = models_dir / "gru_meta_fd001.json"

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        "model": "GRU",
        "seq_len": SEQ_LEN,
        "rul_cap": RUL_CAP,
        "feature_cols": FEATURE_COLS,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "model_file": "gru_rul_fd001.pt",
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nGRU training complete.")
    print("Saved model to:", model_path)
    print("Saved scaler to:", scaler_path)
    print("Saved metadata to:", meta_path)


if __name__ == "__main__":
    main()