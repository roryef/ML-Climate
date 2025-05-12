import os
import pandas as pd
from config import RAW_SNOTEL_DIR

# Loads a single SNOTEL CSV
def load_single_station_csv(filepath):
    """
    Loads a single SNOTEL CSV file, skips metadata, returns cleaned DataFrame.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find line that starts tabular data
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Date,"):
            header_idx = i
            break

    if header_idx is None:
        print(f"Skipping file with no tabular data: {filepath}")
        return None

    # Read tabular data only
    df = pd.read_csv(filepath, skiprows=header_idx)

    # Clean column names
    df.columns = [
        "date",
        "swe_in",
        "precip_accum_in",
        "temp_max_f",
        "temp_min_f",
        "temp_avg_f",
        "precip_increment_in"
    ]

    # Convert data types
    df["date"] = pd.to_datetime(df["date"])
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Extract station ID from filename
    station_id = os.path.basename(filepath).split("_")[0]
    df["station_id"] = station_id
    df["station_name"] = os.path.basename(filepath).split("_", 1)[1].replace(".csv", "")

    return df

# Loads all valid SNOTEL CSVs
def load_all_snotel_data():
    all_dfs = []
    for filename in os.listdir(RAW_SNOTEL_DIR):
        if filename.endswith(".csv"):
            filepath = os.path.join(RAW_SNOTEL_DIR, filename)
            df = load_single_station_csv(filepath)
            if df is not None:
                all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs).sort_values(by=["station_id", "date"])
        print(f"Loaded {len(all_dfs)} stations with {len(combined)} total rows.")
        return combined
    else:
        print("No valid data found.")
        return pd.DataFrame()

if __name__ == "__main__":
    df = load_all_snotel_data()
    print(df.head())

    output_path = os.path.join(RAW_SNOTEL_DIR, "combined_snotel_data.csv")
    df.to_csv(output_path, index=False)
    print(f"💾 Combined data saved to {output_path}")


