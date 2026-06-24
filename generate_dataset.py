"""
generate_dataset.py
Generates a synthetic forest fire dataset with realistic feature distributions.
Saves to data/forest_fire_data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 3000

# --- Weather features ---
temperature   = np.random.uniform(10, 45, N)          # °C
humidity      = np.random.uniform(10, 95, N)           # %
wind_speed    = np.random.uniform(0, 50, N)            # km/h
rainfall      = np.random.exponential(scale=1.5, size=N)  # mm/day

# --- Vegetation / terrain features ---
drought_index    = np.random.uniform(0, 800, N)        # Keetch-Byram Drought Index
ndvi             = np.random.uniform(0.1, 0.9, N)      # Normalized Difference Vegetation Index
slope            = np.random.uniform(0, 45, N)         # degrees
elevation        = np.random.uniform(50, 2500, N)      # meters

# --- Human activity features ---
distance_to_road  = np.random.uniform(0.1, 30, N)      # km
population_density = np.random.uniform(0, 500, N)      # people/km²

# --- Seasonal / temporal ---
month  = np.random.randint(1, 13, N)
is_dry_season = ((month >= 4) & (month <= 9)).astype(int)  # April–September

# --- Fire risk score (domain-rule based) ---
fire_risk = (
    0.30 * (temperature / 45) +
    0.20 * (1 - humidity / 95) +
    0.15 * (wind_speed / 50) +
    0.15 * (drought_index / 800) +
    0.10 * (1 - rainfall / (rainfall.max() + 1e-6)) +
    0.05 * (slope / 45) +
    0.05 * is_dry_season
)

# Add noise & threshold to produce binary label
noise = np.random.normal(0, 0.08, N)
fire_prob = np.clip(fire_risk + noise, 0, 1)
fire_occurred = (fire_prob >= 0.50).astype(int)

df = pd.DataFrame({
    "temperature":        temperature.round(2),
    "humidity":           humidity.round(2),
    "wind_speed":         wind_speed.round(2),
    "rainfall":           rainfall.round(3),
    "drought_index":      drought_index.round(1),
    "ndvi":               ndvi.round(4),
    "slope":              slope.round(2),
    "elevation":          elevation.round(1),
    "distance_to_road":   distance_to_road.round(2),
    "population_density": population_density.round(1),
    "month":              month,
    "is_dry_season":      is_dry_season,
    "fire_occurred":      fire_occurred
})

Path("data").mkdir(exist_ok=True)
df.to_csv("C:\\Users\\Hassaan Zakria\\Downloads\\files (6)\\forest+fires\\forestfires.csv", index=False)

print(f"Dataset saved  → data/forest_fire_data.csv")
print(f"Total samples  : {len(df)}")
print(f"Fire cases     : {fire_occurred.sum()}  ({fire_occurred.mean()*100:.1f} %)")
print(f"No-fire cases  : {(1-fire_occurred).sum()}  ({(1-fire_occurred).mean()*100:.1f} %)")
