from pathlib import Path
import pandas as pd

DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "order_status" in df.columns:
        df = df[df["order_status"].eq("delivered")].copy()
    required = [c for c in ["order_delivered_customer_date", "order_estimated_delivery_date"] if c in df]
    if required:
        df = df.dropna(subset=required).copy()
    if "order_purchase_timestamp" in df and "order_delivered_customer_date" in df:
        df["lead_time_days"] = (
            df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400
    if "order_delivered_customer_date" in df and "order_estimated_delivery_date" in df:
        df["delay_days"] = (
            df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
        ).dt.total_seconds() / 86400
        df["is_late"] = (df["delay_days"] > 0).astype(int)
        df["is_on_time"] = 1 - df["is_late"]
    if "price" in df and "freight_value" in df:
        df["freight_ratio"] = df["freight_value"] / df["price"].replace(0, pd.NA)
    if "lead_time_days" in df:
        df = df[df["lead_time_days"].ge(0)].copy()
    return df.reset_index(drop=True)

def load_demo_data(path="data/sample_orders.csv"):
    return pd.read_csv(path)

def load_olist_data(data_dir="data/olist"):
    p = Path(data_dir)
    orders = pd.read_csv(p / "olist_orders_dataset.csv")
    items = pd.read_csv(p / "olist_order_items_dataset.csv")
    return orders.merge(
        items.groupby("order_id", as_index=False).agg(
            price=("price", "sum"), freight_value=("freight_value", "sum")
        ),
        on="order_id",
        how="left",
    )
