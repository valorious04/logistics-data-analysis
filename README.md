# Logistics Data Analysis and Route Optimization

A Python-based logistics analytics project for analyzing e-commerce delivery performance, calculating operational KPIs, identifying late-delivery risk, segmenting sellers, and demonstrating route optimization.

## Business Problem
E-commerce logistics teams need to reduce delivery delays and freight cost while improving service reliability. This project provides a reproducible analytics workflow from data cleaning and KPI analysis to predictive modeling, clustering, and route optimization.

## KPIs
- On-Time Delivery Rate
- Late Delivery Rate
- Average Delivery Lead Time
- Freight Cost per Order
- Freight-to-Order Value Ratio

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Google OR-Tools.

## Project Structure
```
data/       sample/demo data and dataset instructions
notebooks/  exploratory notebook
src/        reusable analytics modules
outputs/    generated reports/charts
run_pipeline.py
requirements.txt
```

## Run
Use Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python run_pipeline.py --demo
```

For the public Brazilian E-Commerce Olist dataset, place the CSV files described in `data/README.md` inside `data/olist/` and run:

```bash
python run_pipeline.py --olist
```

## Workflow
Collection → Cleaning → KPI Analysis → EDA → Feature Engineering → Prediction → Seller Clustering → Route Optimization → Decision Support.

## Important Note
The included demo dataset is synthetic for reproducibility. The Olist workflow is designed for public historical data and does not represent live traffic or vehicle telemetry. A production system should use road-network travel times, traffic, fleet capacity, delivery windows, and live operational data.

## Author
Ritesh Raj

## Week 2 - Data Preprocessing

Week 2 adds a reproducible preprocessing pipeline in `src/week2_preprocessing_pipeline.py`.

The pipeline covers:
- schema and data-type validation
- timestamp parsing
- duplicate detection
- business-aware missing-value handling
- delivery lead-time and delay feature engineering
- IQR-based outlier capping
- StandardScaler normalization
- before/after data-quality reporting

Run it after downloading the public Olist dataset and placing
`olist_orders_dataset.csv` in `data/olist/`:

```bash
python src/week2_preprocessing_pipeline.py
```

Outputs are written to `outputs/week2/`.

## Week 2 - Data Preprocessing

Week 2 adds a reproducible preprocessing pipeline for the Olist orders dataset. It covers:
- data-quality profiling and missing-value analysis
- duplicate detection and business-rule validation
- timestamp parsing and logistics feature engineering
- IQR-based outlier capping
- numerical standardization with Scikit-learn StandardScaler
- before/after data-quality reporting

Run the Week 2 pipeline after placing the Olist orders CSV under `data/olist/`:

```bash
python src/week2_preprocessing_pipeline.py
```

Outputs are written to `outputs/week2/`.
