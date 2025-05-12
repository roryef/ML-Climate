# Project Timeline (retroactive)

## April 15 — Initial Project Definition
- Explored multiple directions including wind, soil moisture, and energy before narrowing in on snowpack modeling.
- Identified April 1st snow water equivalent (SWE) as a key prediction target due to its importance for water resource planning.

## April 16–24 — Dataset Investigation and Planning
- Examined station-level metadata and coverage for SNOTEL.
- Analyzed structure and access protocol for PRISM climate datasets.
- Established preliminary workflow: collect → clean → extract → model.

## April 25 — Scope Clarification
- Meeting with Alp
- Narrowed geographic scope to Colorado SNOTEL stations (118 total).
- Set modeling objective as predicting peak April 1st SWE from climate and snowpack variables.
- Decided on monthly temporal resolution due to volume and availability of PRISM data.
- Confirmed that ensemble methods and LSTM would be compared for model performance.

## April 26–May 2 — Data Acquisition, Extraction, Aggregation
- Implemented automated download scripts for both PRISM and SNOTEL data.
- Verified PRISM FTP file structure and updated logic to target monthly `.zip` files for each variable and year.
- Downloaded and organized data for 2000 through 2024.
- Developed extraction logic to unzip PRISM raster files and extract pixel-level values using station coordinates.
- Batched processing across variables (`ppt`, `tmin`, `tmax`) and 25 years.
- Generated a unified CSV of PRISM features by year, station, and season.

## May 3–9 — Feature Engineering, Dataset Merging, Modeling and Evaluation
- Computed derived variables from SNOTEL
- Computed PRISM-based features, including seasonal freeze days and matched climate statistics.
- Ensured temporal alignment across datasets.
- Merged PRISM and SNOTEL feature sets on station ID, name, year, and season.
- Trained and evaluated four models.
- Included residual and prediction plots as well as feature importance outputs.
- Integrated 5-fold CV for ensemble models to estimate generalization error.
  
## May 10–11 — Results Analysis and Report Drafting
- Wrote sections of the report covering introduction, data sources, preprocessing, methodology, results, and discussion.
- Compiled visual outputs and adjusted LaTeX formatting.
- Organized table of results and identified key takeaways from model comparison.
