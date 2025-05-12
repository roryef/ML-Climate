import os
import pandas as pd

# Paths
INPUT_PATH = os.path.join("src", "data", "prism", "processed", "prism_extracted_combined_2000_2024.csv")
OUTPUT_PATH = os.path.join("src", "data", "prism", "processed", "prism_features.csv")

# Load combined PRISM dataset
df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

# Drop rows with no values at all
df = df.dropna(subset=["ppt", "tmin", "tmax"], how="all")

# Generate new features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["season"] = df["month"] % 12 // 3 + 1  # 1 = winter, 2 = spring, etc.

# Climate indices
df["tavg"] = (df["tmax"] + df["tmin"]) / 2
df["freeze_risk"] = df["tmin"].apply(lambda x: 1 if x < 32 else 0)

# Aggregate to seasonal features
agg_df = df.groupby(["station_id", "station_name", "year", "season"]).agg({
    "ppt": "sum",
    "tavg": "mean",
    "tmin": "min",
    "tmax": "max",
    "freeze_risk": "sum"
}).reset_index()

# Rename columns
agg_df = agg_df.rename(columns={
    "ppt": "seasonal_precip_in",
    "tavg": "seasonal_temp_avg_f",
    "tmin": "seasonal_temp_min_f",
    "tmax": "seasonal_temp_max_f",
    "freeze_risk": "seasonal_freeze_days"
})

# Save output
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
agg_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Engineered PRISM features saved to {OUTPUT_PATH}")
