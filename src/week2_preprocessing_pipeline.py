"""
Week 2 - Data Collection, Cleaning and Preprocessing
Logistics Data Analysis Project

This module demonstrates a reproducible preprocessing pipeline for the
Brazilian E-Commerce Public Dataset by Olist.

Expected input:
    data/olist/olist_orders_dataset.csv
    data/olist/olist_order_items_dataset.csv
    data/olist/olist_products_dataset.csv
    data/olist/olist_customers_dataset.csv
    data/olist/olist_sellers_dataset.csv

The code is designed to:
1. Load and validate source data
2. Standardize column names and data types
3. Parse timestamps safely
4. Handle missing values using business-aware rules
5. Remove duplicate order records
6. Detect and cap extreme numeric outliers using IQR
7. Create logistics features such as delivery lead time and delay
8. Normalize selected numerical features using StandardScaler
9. Save a model-ready dataset
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data/olist")
OUTPUT_DIR = Path("outputs/week2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_orders(path: Path) -> pd.DataFrame:
    """Load the Olist orders table."""
    df = pd.read_csv(path)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names consistent and easy to work with."""
    out = df.copy()
    out.columns = (
        out.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return out


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert logistics timestamp columns to datetime."""
    out = df.copy()
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact data-quality report."""
    report = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "unique_count": df.nunique(dropna=True),
    })
    return report.sort_values(["missing_pct", "missing_count"], ascending=False)


def remove_duplicate_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and duplicate order IDs."""
    out = df.drop_duplicates().copy()
    if "order_id" in out.columns:
        out = out.drop_duplicates(subset="order_id", keep="first")
    return out


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply business-aware missing-value handling.

    - Keep missing delivery timestamps for non-delivered orders.
    - For delivered orders, drop rows where actual delivery date is absent
      because delivery-time KPIs cannot be calculated reliably.
    - Fill missing categorical status with 'unknown' only if needed.
    """
    out = df.copy()

    if "order_status" in out.columns:
        out["order_status"] = out["order_status"].fillna("unknown")

    if "order_status" in out.columns and "order_delivered_customer_date" in out.columns:
        delivered = out["order_status"].eq("delivered")
        out = out.loc[
            ~(delivered & out["order_delivered_customer_date"].isna())
        ].copy()

    return out


def add_logistics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lead time, delay, and late/on-time flags."""
    out = df.copy()

    if {
        "order_purchase_timestamp",
        "order_delivered_customer_date",
    }.issubset(out.columns):
        out["lead_time_days"] = (
            out["order_delivered_customer_date"]
            - out["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400

    if {
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    }.issubset(out.columns):
        out["delay_days"] = (
            out["order_delivered_customer_date"]
            - out["order_estimated_delivery_date"]
        ).dt.total_seconds() / 86400
        out["is_late"] = (out["delay_days"] > 0).astype(int)
        out["is_on_time"] = (out["delay_days"] <= 0).astype(int)

    if "lead_time_days" in out.columns:
        out.loc[out["lead_time_days"] < 0, "lead_time_days"] = np.nan

    return out


def iqr_cap(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Cap extreme values using the IQR rule instead of deleting rows."""
    s = series.copy()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return s.clip(lower=lower, upper=upper)


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap selected continuous logistics variables."""
    out = df.copy()
    for col in ["lead_time_days", "delay_days"]:
        if col in out.columns:
            out[col] = iqr_cap(out[col])
    return out


def normalize_features(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize selected numeric features to mean 0 and std 1."""
    out = df.copy()
    available = [c for c in columns if c in out.columns]

    if not available:
        return out, StandardScaler()

    scaler = StandardScaler()
    out[[f"{c}_scaled" for c in available]] = scaler.fit_transform(
        out[available].fillna(out[available].median())
    )
    return out, scaler


def preprocess_orders(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full Week 2 preprocessing pipeline."""
    df = load_orders(path)
    raw_quality = quality_report(df)

    df = standardize_columns(df)
    df = parse_dates(df)
    df = remove_duplicate_orders(df)
    df = handle_missing_values(df)
    df = add_logistics_features(df)
    df = cap_outliers(df)

    df, scaler = normalize_features(
        df,
        ["lead_time_days", "delay_days"]
    )

    final_quality = quality_report(df)
    return df, pd.concat(
        {"before": raw_quality, "after": final_quality},
        axis=1
    )


def main():
    orders_path = DATA_DIR / "olist_orders_dataset.csv"

    if not orders_path.exists():
        print(
            "Olist orders file not found.\n"
            f"Expected: {orders_path}\n"
            "Download the public Olist dataset and place the CSV files in "
            "data/olist/."
        )
        return

    processed, quality = preprocess_orders(orders_path)

    processed.to_csv(OUTPUT_DIR / "preprocessed_orders.csv", index=False)
    quality.to_csv(OUTPUT_DIR / "data_quality_report.csv")

    print("Week 2 preprocessing completed.")
    print(f"Rows after preprocessing: {len(processed):,}")
    print(f"Saved: {OUTPUT_DIR / 'preprocessed_orders.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'data_quality_report.csv'}")


if __name__ == "__main__":
    main()
