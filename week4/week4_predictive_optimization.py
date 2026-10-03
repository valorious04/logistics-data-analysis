import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load simulated logistics data
df = pd.read_csv("week4_logistics_prediction_dataset.csv")
X = df.drop(columns="delivery_days")
y = df["delivery_days"]

categorical = ["region", "carrier", "traffic_level", "priority"]
numeric = ["distance_km", "shipment_volume", "weight_kg"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

models = {
    "Linear Regression": Pipeline([("pre", preprocess), ("model", LinearRegression())]),
    "Random Forest": Pipeline([
        ("pre", preprocess),
        ("model", RandomForestRegressor(
            n_estimators=180, max_depth=10, min_samples_leaf=2,
            random_state=42, n_jobs=-1))
    ])
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name)
    print("MAE:", mean_absolute_error(y_test, pred))
    print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)
    print("R2:", r2_score(y_test, pred))

# 5-fold cross-validation for the Random Forest pipeline
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mae = -cross_val_score(models["Random Forest"], X, y, cv=cv,
                          scoring="neg_mean_absolute_error")
print("5-fold CV MAE:", cv_mae.mean(), "+/-", cv_mae.std())

# Hyperparameter tuning
search = GridSearchCV(
    models["Random Forest"],
    {
        "model__n_estimators": [120, 180],
        "model__max_depth": [8, 10, None],
        "model__min_samples_leaf": [1, 2]
    }, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1
)
search.fit(X_train, y_train)
print("Best parameters:", search.best_params_)

# Optimization concept:
# For the same shipment scenario, predict delivery time for each available carrier
# and choose the lowest-risk feasible carrier subject to cost/capacity/SLA constraints.
scenario = X_test.iloc[[0]].copy()
for carrier in ["Carrier_A", "Carrier_B", "Carrier_C"]:
    scenario["carrier"] = carrier
    print(carrier, search.best_estimator_.predict(scenario)[0])
