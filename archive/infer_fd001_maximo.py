import os
import json
from datetime import datetime

import numpy as np
import torch

from preprocess import load_data, scale_features
from model import LSTMRegressor

SEQ_LEN = 30
RUL_CAP = 130
THRESHOLD = 30  # create work order if predicted RUL < this


def load_rul_file(path):
    with open(path, "r") as f:
        vals = [int(line.strip()) for line in f if line.strip()]
    return np.array(vals, dtype=np.float32)


def last_window_per_unit(df, feature_cols, seq_len=30):
    X, units = [], []
    for unit_id, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        if len(g) < seq_len:
            continue
        X.append(g[feature_cols].values[-seq_len:])
        units.append(int(unit_id))
    return units, np.array(X, dtype=np.float32)


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    train_path = os.path.join(BASE_DIR, "data", "train_FD001.txt")
    test_path = os.path.join(BASE_DIR, "data", "test_FD001.txt")
    rul_path = os.path.join(BASE_DIR, "data", "RUL_FD001.txt")
    model_path = os.path.join(BASE_DIR, "models", "lstm_rul_fd001.pt")

    feature_cols = ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

    # Load train/test
    train_df = load_data(train_path)
    test_df = load_data(test_path)

    # Fit scaler on train, apply to test (best practice)
    train_df, scaler = scale_features(train_df, feature_cols)
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # One sample per engine: last 30 cycles
    units, X_test = last_window_per_unit(test_df, feature_cols, seq_len=SEQ_LEN)

    # True RUL labels for test engines (one per engine)
    y_true = load_rul_file(rul_path)[: len(units)]

    # Load model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = LSTMRegressor(input_size=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Predict
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().flatten()

    preds = np.clip(preds, 0, RUL_CAP)

    # Basic evaluation
    mse = float(np.mean((preds - y_true) ** 2))
    mae = float(np.mean(np.abs(preds - y_true)))

    print(f"Device: {device}")
    print(f"Test engines: {len(units)}")
    print(f"Test MSE: {mse:.2f}")
    print(f"Test MAE: {mae:.2f}")

    # Generate Maximo-style work orders
    work_orders = []
    for unit_id, pred_rul in zip(units, preds):
        if pred_rul < THRESHOLD:
            work_orders.append(
                {
                    "asset_id": f"ENGINE-{unit_id:03d}",
                    "predicted_rul_cycles": float(pred_rul),
                    "maintenance_type": "Preventive",
                    "priority": "High",
                    "issue": "Predicted low remaining useful life",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "integration_note": "Simulated payload for IBM Maximo EAM interface",
                }
            )

    out_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "work_orders_fd001.json")

    with open(out_path, "w") as f:
        json.dump(work_orders, f, indent=2)

    print(f"Work orders created: {len(work_orders)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
