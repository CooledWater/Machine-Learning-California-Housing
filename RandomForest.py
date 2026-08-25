from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error

from src.Preprocessing import preprocessing
from src.TrainTestSplit import housing, housing_y


forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
# forest_rmses = -cross_val_score(forest_reg, housing, housing_y, scoring="neg_root_mean_squared_error", cv=10)

forest_reg.fit(housing, housing_y) # takes about 20 seconds for one run

forest_predictions = forest_reg.predict(housing)

forest_rmse = root_mean_squared_error(housing_y, forest_predictions)
print("Random forest rmse =", forest_rmse) 

# forest_rmse = 17519.685029292894
# this is the rmse on training set, which is lower than rmse on validation set
# meaning there is overfitting to the training set