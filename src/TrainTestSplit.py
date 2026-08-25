from src.ReadCSV import housing_full
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

# create income categories
housing_full["income_cat"] = pd.cut(housing_full["median_income"], 
                                    bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
                                    labels=[1, 2, 3, 4, 5])

# splitting with stratified sampling 
train_1, test_1 = train_test_split(housing_full, test_size=0.2, random_state=42, stratify=housing_full["income_cat"])

# now income_cat is useless, we drop it 
for set in (train_1, test_1): 
    set.drop("income_cat", axis=1, inplace=True)

# make a copy of train_1 for future use 
housing = train_1.drop("median_house_value", axis=1)
housing_y = train_1["median_house_value"].copy()