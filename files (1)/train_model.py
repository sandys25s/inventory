"""
Inventory Demand Forecasting - ML Model Training
Trains a Random Forest Regressor on synthetic inventory data
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─── Generate Realistic Synthetic Dataset ──────────────────────────────────────
N = 5000
categories = ['Electronics', 'Clothing', 'Food & Beverage', 'Home & Garden', 'Sports', 'Toys', 'Books', 'Health & Beauty']
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
promo_types = ['None', 'Discount', 'Bundle', 'BOGO', 'Flash Sale']

data = {
    'product_category': np.random.choice(categories, N),
    'month': np.random.randint(1, 13, N),
    'day_of_week': np.random.randint(0, 7, N),
    'season': np.random.choice(seasons, N),
    'price': np.random.uniform(5, 500, N),
    'discount_pct': np.random.uniform(0, 50, N),
    'promotion_type': np.random.choice(promo_types, N),
    'stock_level': np.random.randint(0, 500, N),
    'lead_time_days': np.random.randint(1, 30, N),
    'supplier_reliability': np.random.uniform(0.5, 1.0, N),
    'historical_avg_demand': np.random.uniform(10, 300, N),
    'demand_trend': np.random.uniform(-0.3, 0.3, N),     # % change trend
    'competitor_price_ratio': np.random.uniform(0.7, 1.3, N),
    'customer_rating': np.random.uniform(1, 5, N),
    'num_reviews': np.random.randint(0, 2000, N),
    'reorder_point': np.random.randint(10, 150, N),
    'is_holiday_week': np.random.randint(0, 2, N),
    'region_demand_index': np.random.uniform(0.5, 2.0, N),
}

df = pd.DataFrame(data)

# ─── Construct Target: demand units (with realistic non-linear relationships) ──
base_demand = df['historical_avg_demand'].copy()

# Category multipliers
cat_mult = {'Electronics': 1.2, 'Clothing': 1.1, 'Food & Beverage': 1.5,
            'Home & Garden': 0.9, 'Sports': 1.0, 'Toys': 0.8, 'Books': 0.7, 'Health & Beauty': 1.3}
df['cat_mult'] = df['product_category'].map(cat_mult)

# Seasonal effect
season_mult = {'Spring': 1.05, 'Summer': 1.2, 'Fall': 1.1, 'Winter': 1.3}
df['season_mult'] = df['season'].map(season_mult)

# Promotion effect
promo_mult = {'None': 1.0, 'Discount': 1.3, 'Bundle': 1.15, 'BOGO': 1.4, 'Flash Sale': 1.6}
df['promo_mult'] = df['promotion_type'].map(promo_mult)

# Price sensitivity (higher price = lower demand)
price_sensitivity = 1 - (df['price'] - 5) / (500 * 1.5)

# Holiday boost
holiday_mult = 1 + df['is_holiday_week'] * 0.4

# Trend effect
trend_effect = 1 + df['demand_trend']

# Competitor price ratio (lower = we're cheaper = more demand)
comp_effect = 2 - df['competitor_price_ratio']

# Rating effect
rating_effect = df['customer_rating'] / 3.0

# Region effect
region_effect = df['region_demand_index']

# Discount effect
discount_effect = 1 + df['discount_pct'] / 100

demand = (base_demand * df['cat_mult'] * df['season_mult'] * df['promo_mult'] *
          price_sensitivity * holiday_mult * trend_effect * comp_effect *
          rating_effect * region_effect * discount_effect)

# Add noise
noise = np.random.normal(0, demand * 0.1)
demand = np.clip(demand + noise, 1, None)

df['demand_units'] = demand.round().astype(int)

# ─── Feature Engineering ────────────────────────────────────────────────────────
df.drop(columns=['cat_mult', 'season_mult', 'promo_mult'], inplace=True)

# Encode categoricals
le_category = LabelEncoder()
le_season = LabelEncoder()
le_promo = LabelEncoder()

df['product_category_enc'] = le_category.fit_transform(df['product_category'])
df['season_enc'] = le_season.fit_transform(df['season'])
df['promotion_type_enc'] = le_promo.fit_transform(df['promotion_type'])

# Save encoders for inference
encoders = {
    'product_category': le_category,
    'season': le_season,
    'promotion_type': le_promo
}

FEATURES = [
    'product_category_enc', 'month', 'day_of_week', 'season_enc',
    'price', 'discount_pct', 'promotion_type_enc', 'stock_level',
    'lead_time_days', 'supplier_reliability', 'historical_avg_demand',
    'demand_trend', 'competitor_price_ratio', 'customer_rating',
    'num_reviews', 'reorder_point', 'is_holiday_week', 'region_demand_index'
]

X = df[FEATURES]
y = df['demand_units']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─── Train Model ────────────────────────────────────────────────────────────────
print("Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)

# ─── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# Feature importances
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop 10 Feature Importances:")
print(importances.head(10))

# ─── Save Artifacts ─────────────────────────────────────────────────────────────
joblib.dump(model, '/home/claude/demand_model.pkl')
joblib.dump(encoders, '/home/claude/encoders.pkl')

metadata = {
    'features': FEATURES,
    'categories': list(le_category.classes_),
    'seasons': list(le_season.classes_),
    'promotion_types': list(le_promo.classes_),
    'metrics': {
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'r2': round(r2, 4),
        'mape': round(mape, 2)
    },
    'feature_importances': {k: round(float(v), 4) for k, v in importances.items()},
    'train_size': len(X_train),
    'test_size': len(X_test)
}

with open('/home/claude/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Model, encoders, and metadata saved!")
