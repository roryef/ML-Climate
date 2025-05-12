# src/merge_prism_snotel.py

import os
import pandas as pd

# Paths
SNOTEL_PATH = os.path.join("src", "data", "snotel", "combined_snotel_data.csv")
PRISM_PATH = os.path.join("src", "data", "prism", "processed", "prism_monthly_extracted.csv")
OUTPUT_PATH = os.path.join("src", "data", "merged", "merged_prism_snotel.csv")

# Load data
snotel_df = pd.read_csv(SNOTEL_PATH, parse_dates=["date"])
prism_df = pd.read_csv(PRISM_PATH, parse_dates=["date"])

# Add year and month for merge alignment
snotel_df["year"] = snotel_df["date"].dt.year
snotel_df["month"] = snotel_df["date"].dt.month

prism_df["year"] = prism_df["date"].dt.year
prism_df["month"] = prism_df["date"].dt.month

# Merge on station, year, and month
merged_df = pd.merge(
    snotel_df,
    prism_df,
    on=["station_id", "station_name", "year", "month"],
    how="inner"
)

# Optional: add seasonal label for further grouping
merged_df["season"] = merged_df["month"] % 12 // 3 + 1

# Save
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
merged_df.to_csv(OUTPUT_PATH, index=False)

print(f"Merged SNOTEL + PRISM monthly dataset saved to {OUTPUT_PATH}")
