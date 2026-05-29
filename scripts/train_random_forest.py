import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.preprocess import load_data, add_rul, scale_features, make_sequences

SEQ_LEN = 30
RUL_CAP = 130
FEATURE_COLS = ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "train_FD001.txt"
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    df = load_data(str(data_path))
    df = add_rul(df)
    df, scaler = scale_features(df, FEATURE_COLS)

    X, y = make_sequences(df, FEATURE_COLS, seq_len=SEQ_LEN, rul_cap=RUL_CAP)
    X_flat = X.reshape(X.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse**0.5

    model_path = models_dir / "random_forest_fd001.joblib"
    joblib.dump(model, model_path)

    print("Random Forest training complete.")
    print(f"RMSE: {rmse:.2f}")
    print("Saved model to:", model_path)


if __name__ == "__main__":
    main()