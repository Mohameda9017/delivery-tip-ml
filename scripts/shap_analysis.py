import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Why SHAP Analysis Is Important
# ---------------------------------------------------------------
# SHAP helps explain how our Gradient Boosting model makes predictions.
# It shows:
#   • which features matter most (global importance)
#   • how each feature pushes a prediction up or down (local explanation)
#
# This turns the model from a “black box” into an interpretable system,
# letting us confirm the model is learning reasonable patterns and
# providing transparency for real-world use.
# ---------------------------------------------------------------


# ================================================
# Load saved model and scaler
# ================================================
model = joblib.load("../models/final_xgboost.pkl")
scaler = joblib.load("../models/final_scaler.pkl")

print("Loaded final XGBoost model and scaler.")


# ================================================
# Load data and preprocess exactly like training
# ================================================
df = pd.read_csv("../data/synthetic/synthetic_delivery_data.csv")

categorical_cols = ['weather', 'time_of_day', 'day_of_week']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define X and y
y = df["tip_percent"]
X = df.drop(columns=["tip_percent", "tip_amount"])

# Scale numerical features
X_scaled = scaler.transform(X)

print("Data prepared for SHAP analysis.")

# ================================================
# SHAP Explanation
# ================================================
print("Computing SHAP values...")

# TreeExplainer is optimized for tree-based models (GB, XGBoost, RF)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)

print("SHAP values computed.")

# ================================================
# Global SHAP Summary Plot
# ================================================
print("Generating summary plot...")

shap.summary_plot(shap_values, X, show=True)
plt.savefig("../models/shap_summary_plot.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: shap_summary_plot.png")

# ================================================
# SHAP Bar Plot (easiest to read for reports)
# ================================================
print("Generating bar plot...")

shap.summary_plot(shap_values, X, plot_type="bar", show=True)
plt.savefig("../models/shap_bar_plot.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: shap_bar_plot.png")

# ================================================
# SHAP Dependence Plot (for a single feature)
# ================================================
# Choose the top feature for deeper analysis
top_feature = X.columns[0]  # or pick manually

print(f"Generating dependence plot for: {top_feature}")

shap.dependence_plot(
    top_feature,
    shap_values,
    X,
    show=False
)

plt.savefig(f"../models/shap_dependence_{top_feature}.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: shap_dependence_{top_feature}.png")

print("\nSHAP analysis complete.")
