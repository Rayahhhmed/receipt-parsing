import pandas as pd
import os

dictionary = pd.read_csv("training_data/groceries_cleaned.csv")["Product_Name"].to_list()
if "sammi" in dictionary:
    print("yes")
print(len(dictionary))