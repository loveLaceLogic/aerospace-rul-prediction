import os
import torch
import json
from datetime import datetime

from src.preprocess import (
    load_data,
    add_rul,
    scale_features,
    make_sequences,
)
from src.model import LSTMRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PATH = os.path.join(BASE_DIR, "data", "train_FD001.txt")  # using train for demo

feature_cols = ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

# Load + preprocess
df = load_data(TEST_PATH)
df = add_rul(df)
df, scaler = scale_features(df, feature_cols)

X, y = make_sequences(df, feature_cols, seq_len=30, rul_cap=130)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = LSTMRegressor(input_size=len(feature_cols))
model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "models", "lstm_rul_fd001.pt"), map_location=device
    )
)
model.to(device)
model.eval()

# Predict first 10 sequences
X_tensor = torch.tensor(X[:10], dtype=torch.float32).to(device)

with torch.no_grad():
    preds = model(X_tensor).cpu().numpy().flatten()

work_orders = []

for i, pred in enumerate(preds):
    if pred < 30:  # threshold
        work_orders.append(
            {
                "asset_id": f"ENGINE_{i+1}",
                "predicted_RUL": float(pred),
                "maintenance_type": "Preventive",
                "priority": "High",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

print(json.dumps(work_orders, indent=2))
