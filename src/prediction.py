import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = ["price", "freight_value", "product_weight_g", "freight_ratio"]

def train_late_delivery_model(df: pd.DataFrame):
    available = [c for c in FEATURES if c in df.columns]
    model_df = df[available + ["is_late"]].dropna()
    X = model_df[available]
    y = model_df["is_late"]
    if y.nunique() < 2 or len(model_df) < 10:
        return None, {"message": "Not enough data/classes for a model."}
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, prob), 4),
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }
    return model, metrics
