import os

# === Base Directory (src/)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# === Raw Data Directories
RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw_data')
RAW_SNOTEL_DIR = os.path.join(RAW_DATA_DIR, 'snotel')
RAW_PRISM_DIR = os.path.join(RAW_DATA_DIR, 'prism')
RAW_MODIS_DIR = os.path.join(RAW_DATA_DIR, 'modis')

# === Target Forecast Settings
TARGET_VARIABLE = 'swe_in'
TARGET_DATE = '04-01' # MM-DD
TEST_YEAR = 2023

# === Feature Engineering Parameters
LAG_DAYS = [1, 7, 14]
ROLLING_WINDOWS = [3, 7]

# === ML Model Parameters
RANDOM_SEED = 42
N_ESTIMATORS = 100
MAX_DEPTH = 10
TEST_SIZE = 0.2

# === Date Formats
DATE_FMT = "%Y-%m-%d"
DATETIME_FMT = "%Y-%m-%d %H:%M:%S"

