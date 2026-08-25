import joblib
import pandas as pd
import pathlib
from pathlib import Path

df_path = Path("data/data-frame/housing_full.pkl")
if df_path.exists(): 
    housing_full = joblib.load(df_path)
    print("housing_full is loaded from previous runs. \n")
else: 
    housing_full = pd.read_csv(Path("data/housing.csv"))
    df_path.parent.mkdir(parents=True, exist_ok=True) # create parent folders
    df_path.touch() # create an empty file 
    joblib.dump(housing_full, df_path)
    print("housing_full is created. \n")

if __name__ == "__main__": 
    print(housing_full.head())
    # this part only runs if: 
    # python -m src.ReadCSV

    # does not run when it is being imported 