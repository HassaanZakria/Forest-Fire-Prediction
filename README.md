# 🔥 Forest Fire Prediction — Machine Learning Project

A complete end-to-end ML project that predicts **forest fire occurrence (classification)**
using classical machine learning algorithms.

---

## 🚀 How to Run

### Step 1 — Install dependencies

```bash
pip install scikit-learn xgboost pandas numpy matplotlib seaborn
```

### Step 2 — Generate the dataset

```bash
python generate_dataset.py
```

### Step 3 — Train models

```bash
python train.py
```

### Step 4 — Predict on new samples

```bash
# Demo mode (3 built-in scenarios)
python predict.py

# Interactive mode (enter your own values)
python predict.py --interactive
```

---

## 🌿 Features Used

| Feature              | Description                              | Unit         |
|----------------------|------------------------------------------|--------------|
| temperature          | Ambient air temperature                  | °C           |
| humidity             | Relative humidity                        | %            |
| wind_speed           | Wind speed                               | km/h         |
| rainfall             | Daily rainfall                           | mm           |
| drought_index        | Keetch-Byram Drought Index (KBDI)        | 0–800        |
| ndvi                 | Vegetation density (satellite)           | 0.1–0.9      |
| slope                | Terrain slope                            | degrees      |
| elevation            | Altitude above sea level                 | meters       |
| distance_to_road     | Distance to nearest road                 | km           |
| population_density   | People per km²                           | persons/km²  |
| month                | Month of year                            | 1–12         |
| is_dry_season        | Whether in dry season (Apr–Sep)          | 0 / 1        |

---

## 🤖 Models Compared

| Model               | Type                  |
|---------------------|-----------------------|
| Logistic Regression | Linear                |
| Decision Tree       | Tree-based            |
| Random Forest       | Ensemble (Bagging)    |
| Gradient Boosting   | Ensemble (Boosting)   |
| XGBoost             | Ensemble (Boosting)   |

Evaluation uses **5-fold stratified cross-validation (ROC-AUC)** + test-set metrics.
The best model is automatically saved to `models/best_model.pkl`.

---

## 📊 Output Plots

- **model_comparison.png** — Side-by-side AUC comparison of all models
- **roc_curves.png** — ROC curves for all models on the test set
- **confusion_matrix.png** — True/False Positives & Negatives for best model
- **feature_importance.png** — Which features drive fire prediction the most

---

## 📌 Notes

- Dataset is **synthetic** but built with realistic domain-based rules.
- To use **real data**, replace `data/forest_fire_data.csv` with the
  [UCI Forest Fire Dataset](https://archive.ics.uci.edu/ml/datasets/forest+fires)
  or NASA FIRMS fire data, and update the feature columns accordingly.
- The model pipeline includes `StandardScaler`, so no manual scaling is needed at inference.
