import sklearn
import pandas as pd
from pathlib import Path



housing_full = pd.read_csv(Path("datasets/housing/housing.csv"))
print("Data loaded. \n")
