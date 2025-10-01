import os
import numpy as np
import pandas as pd
from config import RAW_SNOTEL_DIR, LAG_DAYS, ROLLING_WINDOWS

# Loads preprocessed data
def load_clean_data():
    filepath = os.path.join(RAW_SNOTEL_DIR, "combined_snotel_data.csv")
    return pd.read_csv(filepath, parse_dates=["date"])

# Adds lagged versions of selected columns
def add_lag_features(df, lag_columns=["swe_in"], lag_days=[1, 7, 14]):
    df = df.sort_values(by=["station_id", "date"])
    for col in lag_columns:
        for lag in lag_days:
            df[f"{col}_lag_{lag}"] = (
                df.groupby("station_id")[col]
                .shift(lag)
                .astype(float)
            )
    return df

# Adds rolling means of selected columns
def add_rolling_features(df, roll_columns=["temp_avg_f", "precip_increment_in"], windows=[3, 7]):
    df = df.sort_values(by=["station_id", "date"])
    for col in roll_columns:
        for win in windows:
            df[f"{col}_roll_{win}"] = (
                df.groupby("station_id")[col]
                .rolling(window=win, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
    return df

# Adds day-of-year, month, year, and cyclic seasonal features
def add_temporal_features(df):
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df

# Adds target variable: SWE on April 1 of each year, per station
def add_april_1_target(df):
    april1_df = df[df["date"].dt.month.eq(4) & df["date"].dt.day.eq(1)]
    april1_targets = april1_df[["station_id", "year", "swe_in"]].rename(columns={"swe_in": "target_april1_swe"})

    # Merge with all dates before April 1
    df = df.merge(april1_targets, on=["station_id", "year"], how="left")
    return df

# Loads data, applies all transformations, and returns final dataset
def generate_features():
    df = load_clean_data()
    df = add_lag_features(df, lag_days=LAG_DAYS)
    df = add_rolling_features(df, windows=ROLLING_WINDOWS)
    df = add_temporal_features(df)
    df = add_april_1_target(df)

    # Filter out rows with missing predictors or targets
    df = df.dropna(subset=["swe_in_lag_14", "target_april1_swe"])

    print(f"Final dataset with target: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

if __name__ == "__main__":
    final_df = generate_features()
    print(final_df.head())

    # Saves to file
    output_path = os.path.join(RAW_SNOTEL_DIR, "snotel_features.csv")
    final_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
