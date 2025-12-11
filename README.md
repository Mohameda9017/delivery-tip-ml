# Delivery Tip Prediction Model

This project builds a machine learning model that predicts **tip percentage** and **tip amount** for food delivery orders based on real-world features such as distance, subtotal, weather, time of day, wait time, and customer behavior.

The project includes:
- Full ML pipeline (training, tuning, evaluation)
- Final model selection using tuned **XGBoost**
- SHAP explainability analysis
- CLI prediction script
- Saved scaler + model for deployment

---

## Project Overview

Tipping behavior is inconsistent and depends on multiple factors.  
This project uses regression modeling to estimate how much a customer will tip based on:

- Delivery distance  
- Order subtotal  
- Wait time  
- Weather (clear, rain, snow)  
- Time of day  
- Day of week  
- Customer communication rating  
- Item count  
- Number of messages sent  

Categorical variables are one-hot encoded, and numerical variables are standardized.

---

## Model Performance

We trained multiple models and evaluated each using 5-fold cross-validation.

| Model | CV MAE | Test MAE |
|-------|--------|----------|
| Linear Regression | 2.814 | — |
| Ridge Regression | 2.814 | — |
| Lasso Regression | 2.830 | — |
| Random Forest | 2.601 | — |
| Gradient Boosting | 2.534 | 2.372 |
| KNN Regressor | 2.985 | — |
| **XGBoost (Tuned)** | **2.471** | **2.356** |

### Final chosen model: **Tuned XGBoost**  
### Final error: **MAE ≈ 2.35 percentage points**

This means the model is typically within ~2.35% of the true tip percentage — very strong performance for predicting human tipping behavior.

---

## Explainability with SHAP

SHAP values reveal how each input feature influences predictions.

### Most important features:
- **Order subtotal** (higher subtotal → higher tip %)
- **Communication rating** (better communication → higher tips)
- **Weather (rain/snow)** (bad weather → larger tips)
- **Delivery distance** (longer trips → higher tips)
- **Wait time** (longer waits → lower tips)

Generated plots:
- `shap_summary_plot.png`
- `shap_bar_plot.png`
- `shap_dependence_<feature>.png`

All stored in `/models/`.

---

## Training the Model

To retrain the entire pipeline:

```bash
python3 train_model.py
