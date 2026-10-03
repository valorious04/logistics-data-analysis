from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def create_charts(df, output_dir="outputs/charts"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.histplot(df["lead_time_days"], bins=20, kde=True)
    plt.title("Delivery Lead Time Distribution")
    plt.xlabel("Lead time (days)")
    plt.tight_layout()
    plt.savefig(out / "delivery_lead_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    df["is_late"].value_counts().sort_index().plot(kind="bar")
    plt.xticks([0, 1], ["On time", "Late"], rotation=0)
    plt.title("On-Time vs Late Deliveries")
    plt.tight_layout()
    plt.savefig(out / "on_time_vs_late.png", dpi=150)
    plt.close()
