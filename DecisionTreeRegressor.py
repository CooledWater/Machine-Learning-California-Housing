from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error
from src.TrainTestSplit import housing, housing_y
from src.Preprocessing import preprocessing


tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_y)
tree_predictions = tree_reg.predict(housing)

tree_error = root_mean_squared_error(housing_y, tree_predictions)
print("Decision tree regressor rmse = ", tree_error)