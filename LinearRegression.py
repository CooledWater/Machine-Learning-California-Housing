from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from src.TrainTestSplit import housing, housing_y
from src.Preprocessing import preprocessing

lin_pipeline = make_pipeline(preprocessing, LinearRegression())
lin_pipeline.fit(housing, housing_y)
housing_predictions = lin_pipeline.predict(housing)
print("Housing predictions: ", housing_predictions[:5].round(-2))
print("Housing labels: ")
print(housing_y.iloc[:5])


# measure performance 
from sklearn.metrics import root_mean_squared_error

lin_error = root_mean_squared_error(housing_y, housing_predictions)
print("rmse = ", lin_error)