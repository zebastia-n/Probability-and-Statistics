import pandas as pd
import numpy as np
import scipy.stats as sci

data = pd.read_csv("housing.csv")
print(data["total_bedrooms"].unique())

