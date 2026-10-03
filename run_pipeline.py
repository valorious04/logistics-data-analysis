import argparse
from pathlib import Path
import pandas as pd

from src.data_cleaning import load_demo_data, load_olist_data, clean_orders
from src.kpi_analysis import calculate_kpis
from src.exploratory_analysis import create_charts
from src.prediction import train_late_delivery_model
from src.clustering import segment_sellers
from src.route_optimization import solve_demo_route

def main():
    parser = argparse.ArgumentParser(description="Logistics analytics pipeline")
    parser.add_argument("--demo", action="store_true", help="run on synthetic demo data")
    parser.add_argument("--olist", action="store_true", help="run on data/olist")
    args = parser.parse_args()

    if args.olist:
        raw = load_olist_data()
    else:
        raw = load_demo_data()

    df = clean_orders(raw)
    Path("outputs").mkdir(exist_ok=True)
    df.to_csv("outputs/cleaned_orders.csv", index=False)

    kpis = calculate_kpis(df)
    pd.DataFrame([kpis]).to_csv("outputs/kpi_summary.csv", index=False)
    print("KPIs:")
    for key, value in kpis.items():
        print(f"  {key}: {value}")

    create_charts(df)
    _, metrics = train_late_delivery_model(df)
    print("\nPrediction metrics:", metrics)
    segments = segment_sellers(df)
    if not segments.empty:
        segments.to_csv("outputs/seller_segments.csv", index=False)

    print("\nDemo route:", solve_demo_route())
    print("\nPipeline completed.")

if __name__ == "__main__":
    main()
