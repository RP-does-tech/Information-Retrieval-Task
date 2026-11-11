import pandas as pd

df = pd.read_csv("data/processed_resources.csv")

selected = df[df["id"].astype(str).isin(["7", "17"])]

for _, row in selected.iterrows():
    print("\nID:", row["id"])
    print("Title:", row["title"])
    print("Topic:", row["topic"])
    print("Summary:", row["summary"])
    print("-" * 50)


