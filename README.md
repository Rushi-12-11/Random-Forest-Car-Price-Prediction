# 🚗 Car Price Prediction using Random Forest

This project is a machine learning regression pipeline built to predict the **selling price of used cars** based on key features such as car age, fuel type, transmission type, and kilometers driven.

I used a **Random Forest Regressor** with proper data preprocessing, feature engineering, hyperparameter tuning, and evaluation. The model was optimized using **RandomizedSearchCV**, and the results were interpreted using both plots and feature importance.

---

## 🧠 Problem Statement

Predicting used car prices can be unpredictable — sellers often overprice, buyers often underquote. The goal is to use machine learning to build a model that can estimate a car’s fair selling price based on historical data and technical specifications.

---

## 📂 Project Structure
```
Random-Forest-Car-Price-Prediction/
├── Image/ # Folder for saved plots (e.g., Actual vs Predicted)
│ └── Actual_vs_PredictedPrice.png
│
├── Notebook/ # Jupyter Notebooks
│ └── car_price_rf.ipynb
│
├── README.md # Project documentation
├── requirments.txt # Project dependencies (spelling corrected!)

```
---

## 📊 Dataset Overview

The dataset includes:
- `year`: Year of manufacture
- `km_driven`: Total kilometers driven
- `fuel`, `seller_type`, `transmission`, `owner`: Categorical metadata
- `selling_price`: Target variable
- `name`: Vehicle brand/model (dropped in this version)

---

## 🛠 Feature Engineering

✔️ **New Features**:
- `car_age` = 2025 - `year`
- `km_per_year` = km_driven / age (log-transformed)
- `price_per_km` = price / km_driven (log-transformed)

✔️ **Preprocessing**:
- Removed extreme price outliers (top 1%)
- Log-transformed `km_driven` to reduce skew

✔️ **Encoding**:
- One-hot encoding using `ColumnTransformer`
- Dropped first category to prevent multicollinearity

✔️ **Target Transformation**:
- Used `log1p()` for target (`selling_price`)
- Inverse transformed predictions using `expm1()`

---

## 🌲 Model Details

- **Algorithm**: Random Forest Regressor
- **Tuning**: `RandomizedSearchCV` with 3-fold CV
- **Key hyperparameters**:
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
  - `max_features`

---

## 📈 Evaluation Metrics

Model performance after inverse transformation:

- **R² Score**
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)

---

## 📉 Sample Results

| Metric | Value (approx.) |
|--------|-----------------|
| R²     | ~0.85            |
| RMSE   | Depends on test set |
| MAE    | Depends on test set |

📊 Visualizations included:
- Actual vs Predicted plot
- Top feature importances

---

## 🧠 Lessons Learned

- End-to-end preprocessing matters more than model choice
- Log-transformation can stabilize noisy regression targets
- `RandomizedSearchCV` is efficient for tuning tree-based models
- Feature engineering can improve model performance significantly

---

## 🚀 What's Next

- [ ] Convert pipeline using `Pipeline` and `ColumnTransformer`
- [ ] Add SHAP for interpretability
- [ ] Compare with XGBoost & LightGBM
- [ ] Deploy using Streamlit or FastAPI
- [ ] Push clean version to GitHub with `joblib` export

---

## 💬 Author

Made with ❤️ by **[Rushikesh Thokade]** — Final Year CSE student passionate about AI/ML and solving real-world problems with code.

📫 Connect on [LinkedIn](www.linkedin.com/in/rushikesh-thokade)  
🔗 Explore more projects on [GitHub](https://github.com/Rushi-12-11)

---

## 📌 Disclaimer

This is a learning project. The model is trained on publicly available data and should not be used for real-world pricing decisions without proper validation.

