# src/merge_prism_snotel.py

import os
import pandas as pd

# Paths
SNOTEL_PATH = os.path.join("src", "data", "snotel", "combined_snotel_data.csv")
PRISM_PATH = os.path.join("src", "data", "prism", "processed", "prism_features.csv")
OUTPUT_PATH = os.path.join("src", "data", "merged", "merged_prism_snotel.csv")

# Load data
snotel_df = pd.read_csv(SNOTEL_PATH, parse_dates=["date"])
prism_df = pd.read_csv(PRISM_PATH)

# Create seasonal metadata
snotel_df["year"] = snotel_df["date"].dt.year
snotel_df["month"] = snotel_df["date"].dt.month
snotel_df["season"] = snotel_df["month"] % 12 // 3 + 1

# Assign each day to the first of the month
snotel_df["month_start"] = snotel_df["date"].values.astype('datetime64[M]')

# Aggregate SNOTEL to seasonal level (keeping month_start for clarity)
snotel_seasonal = snotel_df.groupby(["station_id", "station_name", "year", "season", "month_start"]).agg({
    "swe_in": "max",
    "precip_accum_in": "max",
    "precip_increment_in": "sum",
    "temp_avg_f": "mean",
    "temp_max_f": "max",
    "temp_min_f": "min"
}).reset_index()

# Rename for clarity
snotel_seasonal = snotel_seasonal.rename(columns={
    "swe_in": "peak_swe_in",
    "precip_accum_in": "seasonal_precip_accum_in",
    "precip_increment_in": "seasonal_precip_increment_in",
    "temp_avg_f": "snotel_temp_avg_f",
    "temp_max_f": "snotel_temp_max_f",
    "temp_min_f": "snotel_temp_min_f",
    "month_start": "date"
})

# Merge with PRISM
merged_df = pd.merge(
    snotel_seasonal,
    prism_df,
    on=["station_id", "station_name", "year", "season"],
    how="inner"
)

# Save
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
merged_df.to_csv(OUTPUT_PATH, index=False)
print(f"Merged SNOTEL + PRISM seasonal dataset saved to {OUTPUT_PATH}")
