import os
import zipfile
import pandas as pd
import rasterio
from datetime import datetime

# Paths
RAW_PRISM_DIR = os.path.join("src", "data", "prism")
STATION_META_PATH = os.path.join("src", "data", "snotel", "snotel_metadata.csv")
PROCESSED_DIR = os.path.join("src", "data", "prism", "processed")
COMBINED_OUTPUT = os.path.join(PROCESSED_DIR, "prism_extracted_combined_2000_2024.csv")

# Load and clean station metadata
station_df = pd.read_csv(STATION_META_PATH)
station_df.columns = [col.strip().lower().replace(" ", "_") for col in station_df.columns]

# Validate station metadata
required_cols = {"station_id", "station_name", "latitude", "longitude"}
if not required_cols.issubset(station_df.columns):
    raise ValueError(f"Missing columns in metadata: {required_cols - set(station_df.columns)}")

# Prepare station coordinates
station_coords = [
    (row["station_id"], row["station_name"], row["latitude"], row["longitude"])
    for _, row in station_df.iterrows()
]

# Create output directory
os.makedirs(PROCESSED_DIR, exist_ok=True)
all_year_dfs = []

# Loop through years
for year in range(2000, 2025):
    print(f"\nProcessing year: {year}")
    year_records = []

    for variable in ["ppt", "tmin", "tmax"]:
        var_dir = os.path.join(RAW_PRISM_DIR, variable, str(year))
        if not os.path.exists(var_dir):
            print(f"Directory not found: {var_dir}")
            continue

        # Unzip all ZIPs in the folder
        for zip_file in sorted(f for f in os.listdir(var_dir) if f.endswith(".zip")):
            zip_path = os.path.join(var_dir, zip_file)
            bil_file = zip_file.replace(".zip", ".bil")
            bil_path = os.path.join(var_dir, bil_file)

            if not os.path.exists(bil_path):
                print(f"Unzipping {zip_file}...")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(var_dir)

        # Process the BIL files
        for bil_file in sorted(f for f in os.listdir(var_dir) if f.endswith(".bil")):
            try:
                date_str = bil_file.split("_")[4]
                file_year = int(date_str[:4])
                if file_year != year:
                    continue
                date_obj = datetime.strptime(date_str, "%Y%m").replace(day=1)
            except Exception as e:
                print(f"Failed to parse date from {bil_file}: {e}")
                continue

            bil_path = os.path.join(var_dir, bil_file)
            try:
                with rasterio.open(bil_path) as src:
                    for sid, name, lat, lon in station_coords:
                        try:
                            val = list(src.sample([(lon, lat)]))[0][0]
                            year_records.append({
                                "station_id": sid,
                                "station_name": name,
                                "date": date_obj,
                                "variable": variable,
                                "value": val
                            })
                        except Exception as e:
                            print(f"Failed to extract {variable} at {name} ({sid}): {e}")
            except Exception as e:
                print(f"Failed to open {bil_file}: {e}")

    # Convert records to DataFrame
    if year_records:
        df_year = pd.DataFrame(year_records)
        pivot_df = df_year.pivot_table(
            index=["station_id", "station_name", "date"],
            columns="variable",
            values="value"
        ).reset_index()
        out_path = os.path.join(PROCESSED_DIR, f"prism_extracted_{year}.csv")
        pivot_df.to_csv(out_path, index=False)
        all_year_dfs.append(pivot_df)
        print(f"Saved {year} data to {out_path}")
    else:
        print(f"No data extracted for {year}")

# Combine all yearly CSVs
if all_year_dfs:
    combined_df = pd.concat(all_year_dfs, ignore_index=True)
    combined_df.to_csv(COMBINED_OUTPUT, index=False)
    print(f"\nCombined all years into {COMBINED_OUTPUT}")
else:
    print("No yearly data found. Combined CSV not created.")
