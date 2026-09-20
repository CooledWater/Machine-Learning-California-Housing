# Machine Learning Project on Predicting Californian Housing Price
Based on the book _Hands-on Machine Learning_ by Aurelien Geron.  
Exploratory data analysis and data pipeline can be found in study-notes.ipynb.  
Modularisation was performed for better organisation of source code.  

## Outline
- Do a simple baseline linear regression model ✅
- Compare and record the performance of different models ✅
- Do hyperparameter tuning to improve a model ✅
- Measure final performance with test set

## Model comparison
### Linear regression
Linear regression rmse =  68973  
Mean district median housing price = 206334  
Percentage error =  68973 / 206334 = 0.33   

### Decision Tree Regressor
Decision Tree rmse = 0.0  
It is impossible to predict the price perfectly, which means there is severe overfitting to the training set.  
Use a 10-fold cross validation on the decision tree regressor:  
```python
rmses = -cross_val_score(tree_reg, housing, housing_y, scoring="neg_root_mean_squared_error", cv=10)
```
Decision tree mean rmse = 67013  
Percentage difference between decision tree rmse and linear regression rmse is 0.028, and thus there is no significant improvement.  

### Random forest regressor
Now attempt to use random forest to reduce the overfitting of decision tree. RMSE between training label and training prediction is calculated.  
Random forest rmse = 17520  
Looks a lot better than the 68973 from linear regression. However, let's try a 3-fold cross validation:  
Random forest mean rmse on validation sets = 47975  
Since random forest regressor performs a lot worse on validation sets than on training sets, there is still overfitting on the training set. But random forest regressor is still better than the previous two models. This is our most promising model so far.  

## Hyperparameter tuning
Now conduct hyperparameter tuning on the random forest regressor.  
Random forest mean rmse on 3-fold cross validation using training set = 45027. 
We can see a 6% reduction in RMSE. 

## Final performance on test set
Linear regression test rmse = 73149  
Final random forest test rmse = 45325  
We can see a 38% reduction in RMSE in the final model as compared to the baseline linear regression model. 



