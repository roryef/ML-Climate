import os
import requests
from datetime import datetime
from config import RAW_PRISM_DIR

# Configuration
START_YEAR = 2000
END_YEAR = 2024
VARIABLES = ["tmax", "tmin", "ppt"]
BASE_URL = "https://data.prism.oregonstate.edu/monthly"

def download_prism_month(variable, year, month):
    month_str = f"{month:02d}"
    date_tag = f"{year}{month_str}"
    filename = f"PRISM_{variable}_stable_4kmM3_{date_tag}_bil.zip"
    url = f"{BASE_URL}/{variable}/{year}/{filename}"

    out_dir = os.path.join(RAW_PRISM_DIR, variable, str(year))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    if os.path.exists(out_path):
        print(f"Already downloaded: {filename}")
        return

    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(response.content)
            print(f"Saved to {out_path}")
        else:
            print(f"Failed to download {filename}. Status: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

def download_prism_monthly():
    for variable in VARIABLES:
        for year in range(START_YEAR, END_YEAR + 1):
            for month in range(1, 13):
                download_prism_month(variable, year, month)

if __name__ == "__main__":
    download_prism_monthly()


