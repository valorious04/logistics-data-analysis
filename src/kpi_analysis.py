import pandas as pd

def calculate_kpis(df: pd.DataFrame) -> dict:
    total = len(df)
    return {
        "delivered_orders": total,
        "on_time_delivery_rate_pct": round(df["is_on_time"].mean() * 100, 2) if total else 0,
        "late_delivery_rate_pct": round(df["is_late"].mean() * 100, 2) if total else 0,
        "average_lead_time_days": round(df["lead_time_days"].mean(), 2) if total else 0,
        "average_freight_cost": round(df["freight_value"].mean(), 2) if "freight_value" in df else 0,
    }
