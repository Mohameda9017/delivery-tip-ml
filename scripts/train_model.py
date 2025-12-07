import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import joblib




# Load synthetic data we previously generated
# Note: change filepath based on current directory
df = pd.read_csv("../data/synthetic/synthetic_delivery_data.csv")

# Encode categorical features into numeric 0/1 columns
categorical_cols = ['weather', 'time_of_day', 'day_of_week']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define X and y
y = df['tip_percent']
X = df.drop(columns=['tip_percent', 'tip_amount']) # if we dont remove tip_amount, it causes data leakage. 


# Split the features and calculated tip into training and testing data. (80% training 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n" +"=" * 10 + " Data Overview " + "=" * 10)
print(f"Training set: {len(X_train)} entries")
print(f"Testing set: {len(X_test)} entries")

# We must standardize the features because the model will punish features in lasso and ridge regressiosn too harshly
# For linear regressions, standardizing isn't necessary because the main goal is to minimize prediction error
# However, Lasso and Ridge regressions aim to minimize prediction error AND penalize large coefficients
# So if we don't standardize each feature to have mean=0 sd=1, lasso and ridge regressions will unfairly punish features with low ranges
# For example, order_total=50, rating=5, calculates some tip where the coefficient for rating must be far greater than the coefficient for order_total because otherwise rating would have too insignificant of an impact
# Ridge and Lasso Regressions would see this large coefficient for rating and unfairly punish it for being too far from 0.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize a dictionary (key: name of model | value: model) to hold our models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1)
}

# Add nonlinear models to the models dictionary
models.update({
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    ),
    "KNN Regressor": KNeighborsRegressor(
        n_neighbors=7
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
})
# Using 5-Fold Cross-Validation to evaluate each model
# Cross validation allows you to split your training data into k subsets. 
# So then instead of training your model on the same data, you train it on a different of data. 
# this allows you to get a better idea of how well your model generalizes to unseen data. 
print("\n" + "="*10 + " Cross-Validation (5-Fold) " + "="*10)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {} # dictionary that will store the cv score of each model

for name, model in models.items():
    scores = cross_val_score( # runs 5 fold cross validation by training the model 5 times on different subsets of the training data. record 5 MAE scores
        model,
        X_train_scaled,
        y_train,
        cv=kfold,
        scoring="neg_mean_absolute_error"
    )
    cv_mae = -scores.mean() # we take the negative because sklearn returns negative MAE because it wants higher scores to be better
    cv_scores[name] = cv_mae 
    print(f"{name} CV MAE: {cv_mae:.3f}")



# Choose best model based on lowest CV MAE
best_model_name = min(cv_scores, key=cv_scores.get)
print(f"\nBest UNTUNED model based on CV: {best_model_name}") # for now it is Gradient Boosting

# finding the model with the best cross validation score and training it on the full training set
# we are training it on the full training set because cross validation only trains on subsets of the training set
# In simply terms, it takes our 80% training data and splits it into.5 parts and in each run, it uses 4 parts to train and 1 part to validate.
# So after cross validation, we want to retrain the best model on the full 80% training data to maximize performance
best_model = models[best_model_name]
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)
final_test_mae = mean_absolute_error(y_test, y_pred)

print("\n" + "="*10 + " Final Test Evaluation " + "="*10)
print(f"{best_model_name} CV MAE: {cv_scores[best_model_name]:.3f}")
print(f"{best_model_name} Test MAE: {final_test_mae:.3f}")

# How do we know that the parameters we set (alpha=0.1, alpha=1.0, degree=2) are the most optimal?
# We do not. Thus, we can play around with different parameters to see which parameters allow for optimal performance
print()
print("=" * 10 + " Hyperparameter Tuning for Random Forest " + "=" * 10)

# Define RF hyperparameter search space. In total there is 216 different Random Forest models
# we picked these parameters because we wanted a search space that is big enough to find a better model but small enough to finish in reasonable time.
rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

# the gridsearchcv goes thorugh all the different types of combinations and chooses the best hyperparameters with the lowest MAE
rf_grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=rf_params,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,  
    verbose=1
)

# runs the grid search 
rf_grid.fit(X_train_scaled, y_train)

# Extract the best hyperparameters and best CV score
best_rf_model = rf_grid.best_estimator_
best_rf_params = rf_grid.best_params_
best_rf_cv_mae = -rf_grid.best_score_

print(f"Best Random Forest CV MAE: {best_rf_cv_mae:.3f}")

# evaluating the tuned random forest model
rf_y_pred = best_rf_model.predict(X_test_scaled)
rf_test_mae = mean_absolute_error(y_test, rf_y_pred)

print(f"Untuned Random Forest CV MAE: {cv_scores['Random Forest']:.3f}")
print(f"Tuned Random Forest Test MAE: {rf_test_mae:.3f}\n") 




print("\n" + "=" * 10 + " Gradient Boosting Hyperparameter Tuning " + "=" * 10)

gb_params = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 5],
    "subsample": [0.8, 1.0]
}

gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid=gb_params,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)

gb_grid.fit(X_train_scaled, y_train)

best_gb = gb_grid.best_estimator_
best_gb_params = gb_grid.best_params_
best_gb_cv_mae = -gb_grid.best_score_

print(f"Best Gradient Boosting CV MAE: {best_gb_cv_mae:.3f}")

gb_y_pred = best_gb.predict(X_test_scaled)
gb_test_mae = mean_absolute_error(y_test, gb_y_pred)

print(f"Untuned Gradient Boosting CV MAE: {cv_scores['Gradient Boosting']:.3f}")
print(f"Tuned Gradient Boosting Test MAE: {gb_test_mae:.3f}\n")


print()
print("\n" + "=" * 10 + " XGBoost Hyperparameter Tuning " + "=" * 10)

xgb_params = {
    "n_estimators": [200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

xgb_grid = GridSearchCV(
    estimator=XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    ),
    param_grid=xgb_params,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X_train_scaled, y_train)

best_xgb = xgb_grid.best_estimator_
best_xgb_params = xgb_grid.best_params_
best_xgb_cv_mae = -xgb_grid.best_score_

print(f"Best XGBoost CV MAE: {best_xgb_cv_mae:.3f}")

xgb_y_pred = best_xgb.predict(X_test_scaled)
xgb_test_mae = mean_absolute_error(y_test, xgb_y_pred)
print(f"Untuned XGBoost CV MAE: {cv_scores['XGBoost']:.3f}")
print(f"Tuned XGBoost Test MAE: {xgb_test_mae:.3f}\n")

tuned_test_scores = {
    "Tuned Random Forest": rf_test_mae,
    "Tuned Gradient Boosting": gb_test_mae,
    "Tuned XGBoost": xgb_test_mae
}

best_tuned_name = min(tuned_test_scores, key=tuned_test_scores.get)
print(f"\nBest Tuned Model Based on Test MAE: {best_tuned_name}")  # it was xgboost


# Summary of all model performances
summary_data = []

# Untuned models
for model_name in cv_scores:
    model_cv = cv_scores[model_name]
    
    # Only the best untuned model has a Test MAE, but we can compute test MAE for all if you want.
    if model_name == best_model_name:
        model_test = final_test_mae
    else:
        model_test = None
    
    summary_data.append([model_name, model_cv, model_test])

# Tuned models
summary_data.append(["Tuned Random Forest", best_rf_cv_mae, rf_test_mae])
summary_data.append(["Tuned Gradient Boosting", best_gb_cv_mae, gb_test_mae])
summary_data.append(["Tuned XGBoost", best_xgb_cv_mae, xgb_test_mae])

summary_df = pd.DataFrame(summary_data, columns=["Model", "CV MAE", "Test MAE"])

print("\n========== Final Model Comparison ==========")
print(summary_df)


# Once we have found the best model, we need to train it with the best hyperparameters on all data
print("\n========== Training Final Model on All Data ==========")

# Prepare full X and y
full_df = pd.read_csv("../data/synthetic/synthetic_delivery_data.csv")
full_df = pd.get_dummies(full_df, columns=categorical_cols, drop_first=True)

y_full = full_df["tip_percent"]
X_full = full_df.drop(columns=["tip_percent", "tip_amount"])

# Refit the scaler on ALL data
X_full_scaled = scaler.fit_transform(X_full)

# Train final XGBoost model using BEST hyperparameters
final_model = XGBRegressor(
    **best_xgb_params,       # parameters from GridSearchCV
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_full_scaled, y_full)

print("Final XGBoost model trained on all data.")

# Save the model + scaler
joblib.dump(final_model, "../models/final_xgboost.pkl")
joblib.dump(scaler, "../models/final_scaler.pkl")

print("Saved final XGBoost model and scaler.")




# # Test alphas for Ridge and Lasso Regressions
# print("\nTesting Ridge and Lasso Regression alphas:")
# alphas_to_test = [0.01, 0.1, 1.0, 10.0, 100.0]
# best_ridge_alpha = None
# best_ridge_mae = float('inf')
# best_lasso_alpha = None
# best_lasso_mae = float('inf')

# for alpha in alphas_to_test:

#     ridge = Ridge(alpha = alpha)
#     ridge.fit(X_train_scaled, y_train)
#     y_pred_ridge = ridge.predict(X_test_scaled)
#     ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
    
#     if ridge_mae < best_ridge_mae:
#         best_ridge_mae = ridge_mae
#         best_ridge_alpha = alpha

#     print(f"RIDGE alpha = {alpha} | MAE: {ridge_mae:.3f}")

#     lasso = Lasso(alpha = alpha)
#     lasso.fit(X_train_scaled, y_train)
#     y_pred_lasso = lasso.predict(X_test_scaled)
#     lasso_mae = mean_absolute_error(y_test, y_pred_lasso)

#     if lasso_mae < best_lasso_mae:
#         best_lasso_mae = lasso_mae
#         best_lasso_alpha = alpha
    
#     print(f"LASSO alpha = {alpha} | MAE: {lasso_mae:.3f}")

# print(f"\nBest Ridge alpha: {best_ridge_alpha} (MAE: {best_ridge_mae:.3f})")
# print(f"Best Lasso alpha: {best_lasso_alpha} (MAE: {best_lasso_mae:.3f})")

# # Test degrees for polynomial features
# print("\nTesting Polynomial Regression degrees:")
# test_degrees = [2, 3, 4]
# best_poly_degree = None
# best_poly_mae = float('inf')

# for degree in test_degrees:
#     poly = PolynomialFeatures(degree=degree)
#     X_train_poly = poly.fit_transform(X_train_scaled)
#     X_test_poly = poly.transform(X_test_scaled)
    
#     poly_model = LinearRegression()
#     poly_model.fit(X_train_poly, y_train)
#     y_pred_poly = poly_model.predict(X_test_poly)
#     poly_mae = mean_absolute_error(y_test, y_pred_poly)
    
#     print(f"degree={degree} | MAE: {poly_mae:.3f}")
    
#     if poly_mae < best_poly_mae:
#         best_poly_mae = poly_mae
#         best_poly_degree = degree

# print(f"Best Polynomial degree: {best_poly_degree}\n")


# # To capture nonlinear trends, the code below utilizes polynomial features to capture trends along curves instead of along straight lines
# # With polynomial features, we simply create more features with the feature raised to "degree"
# # Essentially, we create more features for the model to work with, adding flexibility resulting in the ability to capture curves
# # It's important to not make the degree too high as this would cause overfitting and unecessary computational cost
# print("=" * 10 + " Polynomial Regression " + "=" * 10)

# # Create polynomial features for model + fit the features
# # Additionally, create a test set to test out model with
# poly = PolynomialFeatures(degree=best_poly_degree)
# X_train_poly = poly.fit_transform(X_train_scaled)
# X_test_poly = poly.transform(X_test_scaled)

# # Print amount of columns before and after adding polynomial features (.shape returns (row,col) number so .shape[1] returns number of columns, or, features)
# print(f"Starting number of features: {X_train_scaled.shape[1]}")
# print(f"Resulting number of features: {X_train_poly.shape[1]}")


# # Once we have our new features, fit a model with the new features, create prediction set, calculate error based on actual data.
# poly_model = LinearRegression()
# poly_model.fit(X_train_poly, y_train)
# y_pred = poly_model.predict(X_test_poly)
# mae_poly = mean_absolute_error(y_test, y_pred)

# print(f"Polynomial Regression MAE: {mae_poly:.3f}")
# if(mae_poly < best_mae):
#     print(f"Polynomial Regression performed better than linear Regression by {best_mae - mae_poly:.3f}")
# else:
#     print(f"Linear Regression performed better than Polynomial Regression by {mae_poly - best_mae:.3f}")

# # ============================================================
# # PCA + Regression models
# # ============================================================
# print("=" * 10 + " PCA-Based Regression Models " + "=" * 10)

# pca_results = {}

# for n in [2, 3, 4]:
#     pca = PCA(n_components=n)
#     X_train_pca = pca.fit_transform(X_train_scaled)
#     X_test_pca = pca.transform(X_test_scaled)

#     model = LinearRegression()
#     model.fit(X_train_pca, y_train)
#     y_pred = model.predict(X_test_pca)
#     mae = mean_absolute_error(y_test, y_pred)
    
#     pca_results[n] = mae
#     print(f"PCA Components = {n} → MAE: {mae:.3f}")

# print()