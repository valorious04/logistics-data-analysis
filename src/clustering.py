import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def segment_sellers(df: pd.DataFrame, n_clusters=3):
    if "seller_id" not in df:
        return pd.DataFrame()
    seller = df.groupby("seller_id").agg(
        order_count=("order_id", "count"),
        avg_freight=("freight_value", "mean"),
        avg_lead_time=("lead_time_days", "mean"),
        late_rate=("is_late", "mean"),
    ).reset_index()
    if len(seller) < n_clusters:
        return seller
    features = ["order_count", "avg_freight", "avg_lead_time", "late_rate"]
    X = StandardScaler().fit_transform(seller[features])
    seller["segment"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X)
    return seller
