import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("week3_logistics_dataset.csv")
df["month"] = pd.to_datetime(df["month"])

print(df.describe())
print("Mean delivery time:", df["delivery_days"].mean())
print("Median delivery time:", df["delivery_days"].median())
print("On-time rate:", df["on_time"].mean() * 100)

cols = ["shipment_volume","distance_km","delivery_days","transport_cost","fuel_cost","warehouse_cost"]
print(df[cols].corr())

plt.hist(df["delivery_days"], bins=16, edgecolor="black")
plt.xlabel("Delivery Time (days)")
plt.ylabel("Shipments")
plt.title("Distribution of Delivery Times")
plt.show()

monthly = df.groupby(df["month"].dt.to_period("M"))["shipment_volume"].mean()
monthly.plot(marker="o", title="Monthly Shipment Volume")
plt.ylabel("Average Shipment Volume")
plt.show()

plt.scatter(df["distance_km"], df["transport_cost"], alpha=0.65)
plt.xlabel("Distance (km)")
plt.ylabel("Transport Cost")
plt.title("Distance vs Transport Cost")
plt.show()

region_otd = df.groupby("region")["on_time"].mean().sort_values() * 100
region_otd.plot(kind="barh", title="On-Time Delivery by Region")
plt.xlabel("On-time delivery (%)")
plt.show()
