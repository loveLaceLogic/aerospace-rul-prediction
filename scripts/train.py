import json
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from src.preprocess import load_data, add_rul, scale_features, make_sequences
from src.model import LSTMRegressor

# =========================
# Configuration
# =========================
SEQ_LEN = 30
RUL_CAP = 130
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
SEED = 42

FEATURE_COLS = ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


# =========================
# Reproducibility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Device selection
# =========================
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =========================
# Dataset wrapper
# =========================
class RULDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# Main training logic
# =========================
def main():

    set_seed(SEED)

    BASE_DIR = Path(__file__).resolve().parents[1]
    data_path = BASE_DIR / "data" / "train_FD001.txt"
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    print("Using device:", get_device())

    # -------------------------
    # Load & preprocess data
    # -------------------------
    df = load_data(str(data_path))
    df = add_rul(df)

    df, scaler = scale_features(df, FEATURE_COLS)
    X, y = make_sequences(df, FEATURE_COLS, seq_len=SEQ_LEN, rul_cap=RUL_CAP)

    dataset = RULDataset(X, y)

    # -------------------------
    # Train / validation split
    # -------------------------
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    device = get_device()
    model = LSTMRegressor(input_size=len(FEATURE_COLS)).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # -------------------------
    # Validation function
    # -------------------------
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

    # -------------------------
    # Training loop
    # -------------------------
    train_losses = []
    val_losses = []

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
        
        train_losses.append(train_mse)
        val_losses.append(val_mse)
        
        print(
            f"Epoch {epoch:02d} | "
            f"Train MSE: {train_mse:.4f} | "
            f"Val MSE: {val_mse:.4f} | "
            f"Val RMSE: {val_rmse:.4f}"
        )

    # -------------------------
    # Save artifacts
    # -------------------------
    model_path = models_dir / "lstm_rul_fd001.pt"
    scaler_path = models_dir / "scaler_fd001.joblib"
    meta_path = models_dir / "meta_fd001.json"

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        "seq_len": SEQ_LEN,
        "rul_cap": RUL_CAP,
        "feature_cols": FEATURE_COLS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "model_file": "lstm_rul_fd001.pt",
        "scaler_file": "scaler_fd001.joblib",
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved model to:", model_path)
    print("Saved scaler to:", scaler_path)
    print("Saved metadata to:", meta_path)
    outputs_dir = BASE_DIR / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM RUL Training Performance")

    plt.legend()
    plt.tight_layout()

    training_curve_path = outputs_dir / "training_curve.png"

    plt.savefig(training_curve_path)
    plt.close()

    print("Saved training curve to:", training_curve_path)
    print("Training complete.")


if __name__ == "__main__":
    main()
