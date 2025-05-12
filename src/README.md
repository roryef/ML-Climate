# SWE Prediction Pipeline (src/)

This directory contains the full source code for a machine learning pipeline to predict April 1st Snow Water Equivalent (SWE) across Colorado using SNOTEL and PRISM climate data from 2000–2024.


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
   - `merge_data.py`: (Earlier script, may be deprecated)

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


## Output

The outputs of the pipeline include:
- `merged_prism_snotel.csv`: Final dataset with aligned features and target SWE values.
- Figures in `figures/`: Predicted vs actual plots, residual distributions, feature importances.
- Console logs summarizing model performance, cross-validation scores, and LSTM training history.


## Notes

- All data is sourced from the NRCS SNOTEL system and PRISM Climate Group (Oregon State University).
- Temporal resolution is monthly, aggregated to seasonal metrics.
- All models predict peak SWE on April 1st at each station-year.

---

## Author

Rory Eastland-Fruit
Machine Learning and Climate
May 2025
