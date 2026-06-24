"""
train.py
Trains and compares multiple classical ML classifiers for forest fire prediction.
Saves the best model to models/best_model.pkl
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ── 0. Directories ──────────────────────────────────────────────────────────
Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv("C:\\Users\\Hassaan Zakria\\Downloads\\files (6)\\forest+fires\\forestfires.csv")
print("=" * 60)
print("  FOREST FIRE PREDICTION — MODEL TRAINING")
print("=" * 60)
print(f"\nDataset shape : {df.shape}")
print(f"Class balance : Fire={df.fire_occurred.sum()} | No-Fire={(df.fire_occurred==0).sum()}\n")

FEATURES = [c for c in df.columns if c != "fire_occurred"]
TARGET   = "fire_occurred"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ── 2. Define models ──────────────────────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    DecisionTreeClassifier(max_depth=8, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=200, max_depth=12,
                                          random_state=42, n_jobs=-1))
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                              learning_rate=0.1, random_state=42))
    ]),
    "XGBoost": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 use_label_encoder=False, eval_metric="logloss",
                                 random_state=42, n_jobs=-1))
    ]),
}

# ── 3. Cross-validation comparison ───────────────────────────────────────────
print("Cross-Validation Results (5-Fold, metric = ROC-AUC)")
print("-" * 60)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:<22}  AUC = {scores.mean():.4f} ± {scores.std():.4f}")

# ── 4. Train all models on full training set ──────────────────────────────────
print("\nTraining all models on full training set...")
trained = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    trained[name] = pipe

# ── 5. Test-set evaluation ────────────────────────────────────────────────────
print("\nTest-Set Evaluation")
print("-" * 60)
test_aucs = {}
for name, pipe in trained.items():
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    test_aucs[name] = auc
    acc    = (y_pred == y_test).mean()
    print(f"\n{'='*4} {name} {'='*(50-len(name))}")
    print(f"  Accuracy : {acc*100:.2f}%   |   ROC-AUC : {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["No Fire", "Fire"]))

# ── 6. Pick best model ────────────────────────────────────────────────────────
best_name = max(test_aucs, key=test_aucs.get)
best_model = trained[best_name]
print(f"\n★  Best Model : {best_name}  (AUC = {test_aucs[best_name]:.4f})")

with open("models/best_model.pkl", "wb") as f:
    pickle.dump({"model": best_model, "features": FEATURES, "name": best_name}, f)
print("   Saved → models/best_model.pkl")

# ── 7. Plots ──────────────────────────────────────────────────────────────────

# 7a. CV comparison bar chart
fig, ax = plt.subplots(figsize=(9, 4))
names   = list(cv_results.keys())
means   = [cv_results[n].mean() for n in names]
stds    = [cv_results[n].std()  for n in names]
colors  = ["#e74c3c" if n == best_name else "#2980b9" for n in names]
bars    = ax.barh(names, means, xerr=stds, color=colors, edgecolor="white",
                  height=0.5, capsize=4)
ax.set_xlabel("ROC-AUC (5-Fold CV)", fontsize=11)
ax.set_title("Model Comparison — Cross-Validation ROC-AUC", fontsize=13, fontweight="bold")
ax.set_xlim(0.5, 1.0)
for bar, val in zip(bars, means):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)
plt.close()
print("   Plot  → outputs/model_comparison.png")

# 7b. Confusion matrix for best model
y_pred_best = best_model.predict(X_test)
cm  = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["No Fire", "Fire"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()
print("   Plot  → outputs/confusion_matrix.png")

# 7c. ROC curves for all models
fig, ax = plt.subplots(figsize=(7, 5))
for name, pipe in trained.items():
    y_prob = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = test_aucs[name]
    lw  = 2.5 if name == best_name else 1.2
    ax.plot(fpr, tpr, lw=lw, label=f"{name} (AUC={auc:.3f})")
ax.plot([0,1],[0,1], "k--", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("outputs/roc_curves.png", dpi=150)
plt.close()
print("   Plot  → outputs/roc_curves.png")

# 7d. Feature importance (best model — works for tree-based)
try:
    clf = best_model.named_steps["clf"]
    importances = clf.feature_importances_
    feat_df = pd.Series(importances, index=FEATURES).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    feat_df.plot(kind="barh", ax=ax, color="#27ae60", edgecolor="white")
    ax.set_title(f"Feature Importances — {best_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("   Plot  → outputs/feature_importance.png")
except AttributeError:
    print("   (Feature importance not available for this model type)")

print("\nTraining complete!\n")
