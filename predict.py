"""
predict.py
Load the saved best model and predict fire risk for new samples.

Usage:
    python predict.py                    # runs built-in demo samples
    python predict.py --interactive      # enter values manually
"""

import argparse
import pickle
import numpy as np
import pandas as pd

MODEL_PATH = "models/best_model.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["features"], bundle["name"]

def predict_samples(model, features, samples: list[dict]):
    df      = pd.DataFrame(samples)[features]
    preds   = model.predict(df)
    probs   = model.predict_proba(df)[:, 1]

    print("\n" + "=" * 60)
    print("  FOREST FIRE PREDICTION RESULTS")
    print("=" * 60)
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        label     = "🔥 FIRE RISK" if pred == 1 else "✅ NO FIRE"
        risk_lvl  = ("CRITICAL" if prob >= 0.80 else
                     "HIGH"     if prob >= 0.60 else
                     "MODERATE" if prob >= 0.40 else "LOW")
        print(f"\nSample {i+1}:")
        for k, v in samples[i].items():
            print(f"  {k:<22}: {v}")
        print(f"  {'─'*40}")
        print(f"  Prediction    : {label}")
        print(f"  Fire Prob.    : {prob*100:.1f}%")
        print(f"  Risk Level    : {risk_lvl}")
    print()

def interactive_mode(features):
    model, features, name = load_model()
    print(f"\nLoaded model : {name}")
    print("Enter feature values for prediction (press Enter for default):\n")

    DEFAULTS = {
        "temperature": 25.0,
        "humidity": 50.0,
        "wind_speed": 15.0,
        "rainfall": 1.0,
        "drought_index": 400.0,
        "ndvi": 0.5,
        "slope": 10.0,
        "elevation": 500.0,
        "distance_to_road": 5.0,
        "population_density": 50.0,
        "month": 6,
        "is_dry_season": 1,
    }

    sample = {}
    for feat in features:
        default = DEFAULTS.get(feat, 0.0)
        raw = input(f"  {feat} [{default}]: ").strip()
        sample[feat] = float(raw) if raw else float(default)

    predict_samples(model, features, [sample])


# ── Demo samples ─────────────────────────────────────────────────────────────
DEMO_SAMPLES = [
    {   # High-risk: hot, dry, windy summer day
        "temperature": 42.0, "humidity": 12.0, "wind_speed": 40.0,
        "rainfall": 0.1, "drought_index": 720.0, "ndvi": 0.25,
        "slope": 30.0, "elevation": 800.0, "distance_to_road": 1.5,
        "population_density": 120.0, "month": 7, "is_dry_season": 1,
    },
    {   # Low-risk: cool, wet, calm autumn day
        "temperature": 14.0, "humidity": 85.0, "wind_speed": 5.0,
        "rainfall": 12.0, "drought_index": 80.0, "ndvi": 0.75,
        "slope": 5.0, "elevation": 200.0, "distance_to_road": 15.0,
        "population_density": 10.0, "month": 11, "is_dry_season": 0,
    },
    {   # Moderate-risk: warm spring day
        "temperature": 28.0, "humidity": 38.0, "wind_speed": 22.0,
        "rainfall": 0.8, "drought_index": 380.0, "ndvi": 0.50,
        "slope": 15.0, "elevation": 600.0, "distance_to_road": 4.0,
        "population_density": 60.0, "month": 5, "is_dry_season": 1,
    },
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forest Fire Predictor")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter custom feature values interactively")
    args = parser.parse_args()

    model, features, name = load_model()
    print(f"\nLoaded model : {name}")
    print(f"Features     : {features}")

    if args.interactive:
        interactive_mode(features)
    else:
        print("\nRunning predictions on 3 demo scenarios...")
        predict_samples(model, features, DEMO_SAMPLES)
