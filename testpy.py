import pandas as pd


data = pd.read_csv("housing.csv")
print(data["ocean_proximity"].unique())