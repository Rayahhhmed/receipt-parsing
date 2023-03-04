import pandas as pd

PATH = "training_data/groceries.csv"

df = pd.read_csv(PATH)
df = df.dropna()
col_names = ["Product_Name", "Category", "Product_Group"]
df.reset_index(drop=True, inplace=True)
df.to_csv("training_data/groceries_cleaned.csv", index=False)

