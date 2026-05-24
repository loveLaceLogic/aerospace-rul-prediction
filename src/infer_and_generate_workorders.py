import json
import os
from datetime import datetime
from pathlib import Path

import torch

from src.preprocess import add_rul, load_data, make_sequences, scale_features
from src.model import LSTMRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PATH = os.path.join(BASE_DIR, "data", "train_FD001.txt")  # demo data

FEATURE_COLS = ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

df = load_data(TEST_PATH)
df = add_rul(df)
df, scaler = scale_features(df, FEATURE_COLS)

X, y = make_sequences(df, FEATURE_COLS, seq_len=30, rul_cap=130)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = LSTMRegressor(input_size=len(FEATURE_COLS))
model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "models", "lstm_rul_fd001.pt"),
        map_location=device,
    )
)
model.to(device)
model.eval()

X_tensor = torch.tensor(X[:25], dtype=torch.float32).to(device)

with torch.no_grad():
    predictions = model(X_tensor).cpu().numpy().flatten()

work_orders = []

for i, rul in enumerate(predictions):
    if rul <= 20:
        priority = "HIGH"
        action = "Immediate inspection required"
    elif rul <= 50:
        priority = "MEDIUM"
        action = "Schedule maintenance soon"
    else:
        priority = "LOW"
        action = "Continue monitoring"

    work_orders.append(
        {
            "asset_id": f"ENGINE_{i + 1}",
            "predicted_rul": round(float(rul), 2),
            "maintenance_type": "Predictive",
            "priority": priority,
            "recommended_action": action,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

outputs_dir = Path(BASE_DIR) / "outputs"
outputs_dir.mkdir(exist_ok=True)

output_path = outputs_dir / "maintenance_alerts.json"

with open(output_path, "w") as f:
    json.dump(work_orders, f, indent=2)

print(json.dumps(work_orders, indent=2))
print(f"\nSaved maintenance alerts to: {output_path}")