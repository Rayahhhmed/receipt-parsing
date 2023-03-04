import pandas as pd

PATH = "training_data/groceries.csv"

df = pd.read_csv(PATH)
df = df.dropna()
col_names = ["Product_Name", "Category", "Product_Group"]
df.reset_index(drop=True, inplace=True)
df = df[col_names].drop_duplicates()
l = df["Product_Name"].drop_duplicates().apply(str.lower)
l.to_csv("training_data/product_names.csv", index=False)
df.to_csv("training_data/groceries_cleaned.csv", index=False)

