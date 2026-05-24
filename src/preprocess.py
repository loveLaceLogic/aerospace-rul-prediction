import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1)

    cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    df.columns = cols
    return df


def add_rul(df):
    max_cycle = df.groupby("unit")["cycle"].max().reset_index()
    max_cycle.columns = ["unit", "max_cycle"]
    df = df.merge(max_cycle, on="unit", how="left")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df = df.drop(columns=["max_cycle"])
    return df


from sklearn.preprocessing import StandardScaler
import numpy as np


def scale_features(df, feature_cols):
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler


def make_sequences(df, feature_cols, seq_len=30, rul_cap=130):
    X, y = [], []

    for unit_id, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        feats = g[feature_cols].values
        rul = g["rul"].values

        # cap RUL (common in CMAPSS to stabilize training)
        rul = np.clip(rul, 0, rul_cap)

        if len(g) < seq_len:
            continue

        for i in range(len(g) - seq_len + 1):
            X.append(feats[i : i + seq_len])
            y.append(rul[i + seq_len - 1])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y
