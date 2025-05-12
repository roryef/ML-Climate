import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from config import RANDOM_SEED, FIGURE_DIR


def load_features():
    filepath = os.path.join("src", "data", "merged", "merged_prism_snotel.csv")
    df = pd.read_csv(filepath, parse_dates=["date"])
    df["target_april1_swe"] = df["peak_swe_in"]
    return df


def train_test_split_by_year(df, test_start=2019):
    train_df = df[df["year"] < test_start]
    test_df = df[df["year"] >= test_start]
    return train_df, test_df


def evaluate(y_true, y_pred, label=""):
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label} RMSE: {rmse:.3f}, R²: {r2:.3f}")
    return rmse, r2


def plot_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r')
    plt.xlabel("Actual SWE (in)")
    plt.ylabel("Predicted SWE (in)")
    plt.title(f"{model_name}: Actual vs Predicted SWE")
    plt.grid(True)
    save_path = os.path.join(FIGURE_DIR, f"{model_name.lower().replace(' ', '_')}_pred_vs_actual.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📈 Saved prediction plot to {save_path}")
    plt.close()


def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=40, kde=True)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.title(f"{model_name}: Residual Distribution")
    plt.grid(True)
    save_path = os.path.join(FIGURE_DIR, f"{model_name.lower().replace(' ', '_')}_residuals.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📉 Saved residual plot to {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names, model_name):
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    top_features = [feature_names[i] for i in sorted_idx]
    top_importance = importance[sorted_idx]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importance[:15], y=top_features[:15])
    plt.title(f"{model_name}: Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, f"{model_name.lower().replace(' ', '_')}_feature_importance.png")
    plt.savefig(save_path)
    print(f"📊 Saved feature importance plot to {save_path}")
    plt.close()


def cross_validate_model(model, X, y, label="Model"):
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    neg_rmse = cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=kf, n_jobs=-1)
    r2 = cross_val_score(model, X, y, scoring="r2", cv=kf, n_jobs=-1)

    print(f"🔄 {label} Cross-Validation (5-Fold):")
    print(f"  RMSE: {(-neg_rmse).mean():.3f} ± {(-neg_rmse).std():.3f}")
    print(f"  R²:   {r2.mean():.3f} ± {r2.std():.3f}")


def prepare_lstm_data(df, feature_cols, timesteps):
    sequences = []
    targets = []

    grouped = df.groupby(["station_id", "year"])
    for _, group in grouped:
        group = group.sort_values("date")
        if len(group) < timesteps:
            continue
        X_seq = group[feature_cols].to_numpy()
        y_val = group["target_april1_swe"].values[-1]
        sequences.append(X_seq[-timesteps:])
        targets.append(y_val)

    return np.stack(sequences), np.array(targets)


def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def train_and_evaluate_models(df):
    feature_cols = [
        col for col in df.columns
        if col not in ["station_id", "station_name", "year", "season", "date",
                       "target_april1_swe", "peak_swe_in"]
        and df[col].dtype in [np.float64, np.int64]
    ]

    train_df, test_df = train_test_split_by_year(df)
    X_train, y_train = train_df[feature_cols], train_df["target_april1_swe"]
    X_test, y_test = test_df[feature_cols], test_df["target_april1_swe"]

    train_mask = X_train.notna().all(axis=1)
    test_mask = X_test.notna().all(axis=1)
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    print(f"📂 Training set: {X_train.shape}, Testing set: {X_test.shape}")

    # === Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    evaluate(y_test, y_pred_lr, label="Linear Regression")
    plot_predictions(y_test, y_pred_lr, "Linear Regression")

    # === Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    evaluate(y_test, y_pred_rf, label="Random Forest")
    plot_predictions(y_test, y_pred_rf, "Random Forest")
    plot_residuals(y_test, y_pred_rf, "Random Forest")
    plot_feature_importance(rf, feature_cols, "Random Forest")
    cross_validate_model(rf, X_train, y_train, label="Random Forest")

    # === XGBoost
    xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
                       colsample_bytree=0.8, random_state=RANDOM_SEED, n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    evaluate(y_test, y_pred_xgb, label="XGBoost")
    plot_predictions(y_test, y_pred_xgb, "XGBoost")
    plot_residuals(y_test, y_pred_xgb, "XGBoost")
    plot_feature_importance(xgb, feature_cols, "XGBoost")
    cross_validate_model(xgb, X_train, y_train, label="XGBoost")

    # === LSTM
    print("🔁 Preparing data for LSTM...")
    train_df_clean = train_df.dropna(subset=feature_cols + ["target_april1_swe"])
    test_df_clean = test_df.dropna(subset=feature_cols + ["target_april1_swe"])
    X_train_lstm, y_train_lstm = prepare_lstm_data(train_df_clean, feature_cols, timesteps=12)
    X_test_lstm, y_test_lstm = prepare_lstm_data(test_df_clean, feature_cols, timesteps=12)

    print(f"📐 LSTM shape: train {X_train_lstm.shape}, test {X_test_lstm.shape}")
    lstm = build_lstm_model(X_train_lstm.shape[1:])
    lstm.fit(
        X_train_lstm, y_train_lstm,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)],
        verbose=1
    )

    y_pred_lstm = lstm.predict(X_test_lstm).flatten()
    evaluate(y_test_lstm, y_pred_lstm, label="LSTM")
    plot_predictions(y_test_lstm, y_pred_lstm, "LSTM")
    plot_residuals(y_test_lstm, y_pred_lstm, "LSTM")


def ensure_dirs():
    os.makedirs(FIGURE_DIR, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    df = load_features()
    train_and_evaluate_models(df)
