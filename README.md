# SWE Prediction Pipeline

This directory contains the full source code for a machine learning pipeline to predict April 1st Snow Water Equivalent (SWE) across Colorado using SNOTEL and PRISM climate data from 2000–2024.


## Directory Structure

`src/` contains the following key components:

- `data/`  
  - `snotel/` — Raw and combined SNOTEL station data  
  - `prism/` — Raw and processed PRISM climate data  
  - `merged/` — Final merged dataset used for modeling  

- `figures/`  
  - Contains output plots for model performance, residuals, and feature importances  

- `config.py`  
  - Centralized configuration for paths and constants  

- `download_snotel.py`  
  - Downloads and processes SNOTEL metadata and daily data  

- `download_prism.py`  
  - Downloads monthly PRISM raster ZIPs from Oregon State's FTP  

- `extract_prism.py`  
  - Extracts raster values at SNOTEL coordinates for all months and variables  

- `feature_engineer_snotel.py`  
  - Aggregates daily SNOTEL data to seasonal statistics  

- `feature_engineer_prism.py`  
  - Aggregates monthly PRISM data to seasonal features  

- `merge_prism_snotel.py`  
  - Joins engineered PRISM and SNOTEL data on station, year, and season  

- `merge_data.py`  
  - (Optional legacy script for earlier merging workflow)  

- `preprocess.py`  
  - General-purpose preprocessing utilities (not actively used)  

- `train.py`  
  - Trains machine learning models and outputs metrics and figures  

- `README.md`  
  - This README file explaining structure and purpose  


## Overview of Pipeline

1. **Download**
   - `download_snotel.py`: Collects station metadata and daily measurements for Colorado SNOTEL sites.
   - `download_prism.py`: Downloads monthly PRISM climate data (TMAX, TMIN, PPT) from 2000–2024.

2. **Extraction & Aggregation**
   - `extract_prism.py`: Unzips and extracts PRISM raster values at SNOTEL station coordinates.
   - `feature_engineer_snotel.py`: Aggregates daily SNOTEL data to seasonal features.
   - `feature_engineer_prism.py`: Aggregates monthly PRISM data to match seasonal resolution.

3. **Merging**
   - `merge_prism_snotel.py`: Merges seasonal PRISM and SNOTEL features, aligning by station, year, and season.

4. **Modeling**
   - `train.py`: Trains Linear Regression, Random Forest, XGBoost, and LSTM models. Outputs:
     - RMSE and R² metrics
     - Predicted vs. Actual SWE plots
     - Residual distributions
     - Feature importance rankings

5. **Config & Outputs**
   - `config.py`: Holds directory paths and configuration variables.
   - `figures/`: Saved output figures used in the final report and presentation.
   - `data/merged/merged_prism_snotel.csv`: Final dataset used for modeling.


## Running

To reproduce the pipeline:

1. Run `download_snotel.py` and `download_prism.py` to fetch all raw data.
2. Run `extract_prism.py` to convert PRISM BIL files into usable station-level values.
3. Run `feature_engineer_snotel.py` and `feature_engineer_prism.py` to compute seasonal metrics.
4. Merge them with `merge_prism_snotel.py`.
5. Train and evaluate models with `train.py`.


## Output

The outputs of the pipeline include:
- `merged_prism_snotel.csv`: Final dataset with aligned features and target SWE values.
- Figures in `figures/`: Predicted vs actual plots, residual distributions, feature importances.
- Console logs summarizing model performance, cross-validation scores, and LSTM training history.


## Notes

- LSTM performance was weaker due to limited monthly granularity. Future improvements may involve switching to daily-level sequence modeling.
- Ensemble models (Random Forest, XGBoost) significantly outperformed linear and sequential baselines.

- All data is sourced from the NRCS SNOTEL system and PRISM Climate Group (Oregon State University).
- Temporal resolution is monthly, aggregated to seasonal metrics.
- All models predict peak SWE on April 1st at each station-year.


## Author

Rory Eastland-Fruit,
Machine Learning and Climate,
May 2025
