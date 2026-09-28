"""
ml_engine.py - Machine Learning & Optimization Engine
Wraps Random Forest model, handles multi-product forecasting, safety stock math,
reorder point calculations, overstock detection, confidence intervals, and safe retraining pipeline.
"""
import os
import json
import joblib
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, 'demand_model.pkl')
BACKUP_MODEL_PATH = os.path.join(BASE, 'demand_model_backup.pkl')
ENCODERS_PATH = os.path.join(BASE, 'encoders.pkl')
METADATA_PATH = os.path.join(BASE, 'model_metadata.json')

class DemandMLEngine:
    def __init__(self):
        self.load_artifacts()

    def load_artifacts(self):
        """Loads model, encoders, and metadata into memory."""
        self.model = joblib.load(MODEL_PATH)
        self.encoders = joblib.load(ENCODERS_PATH)
        with open(METADATA_PATH, 'r') as f:
            self.metadata = json.load(f)
        self.features = self.metadata['features']

    def get_metadata(self):
        return self.metadata

    def predict_single(self, d):
        """
        Exact backward-compatible prediction method matching the original application.
        Accepts dictionary of 18 parameters and produces ML demand prediction + CI + reorder metrics.
        """
        cat_enc = self.encoders['product_category'].transform([d['product_category']])[0]
        season_enc = self.encoders['season'].transform([d['season']])[0]
        promo_enc = self.encoders['promotion_type'].transform([d['promotion_type']])[0]

        feature_vector = pd.DataFrame([[
            cat_enc,
            int(d['month']),
            int(d['day_of_week']),
            season_enc,
            float(d['price']),
            float(d['discount_pct']),
            promo_enc,
            int(d['stock_level']),
            int(d['lead_time_days']),
            float(d['supplier_reliability']),
            float(d['historical_avg_demand']),
            float(d['demand_trend']),
            float(d['competitor_price_ratio']),
            float(d['customer_rating']),
            int(d['num_reviews']),
            int(d['reorder_point']),
            int(d['is_holiday_week']),
            float(d['region_demand_index']),
        ]], columns=self.features)

        predicted_demand = float(self.model.predict(feature_vector)[0])

        fv_vals = feature_vector.values
        all_preds = np.array([est.predict(fv_vals)[0] for est in self.model.estimators_])
        lower_ci = float(np.percentile(all_preds, 10))
        upper_ci = float(np.percentile(all_preds, 90))

        safety_stock_factor = 1 + (upper_ci - lower_ci) / (2 * max(predicted_demand, 1))
        recommended_order = round(predicted_demand * safety_stock_factor)
        reorder_needed = int(d['stock_level']) < int(d['reorder_point'])
        days_of_stock = round(int(d['stock_level']) / max(predicted_demand / 30.0, 0.1))

        return {
            'predicted_demand': round(predicted_demand),
            'lower_ci': round(lower_ci),
            'upper_ci': round(upper_ci),
            'recommended_order_qty': recommended_order,
            'reorder_needed': reorder_needed,
            'days_of_stock_remaining': days_of_stock,
            'confidence_interval_width': round(upper_ci - lower_ci),
        }

    def predict_batch(self, items):
        """
        Batch prediction for scenario comparisons (backward-compatible).
        """
        results = []
        for d in items:
            cat_enc = self.encoders['product_category'].transform([d['product_category']])[0]
            season_enc = self.encoders['season'].transform([d['season']])[0]
            promo_enc = self.encoders['promotion_type'].transform([d['promotion_type']])[0]

            fv = pd.DataFrame([[
                cat_enc, int(d['month']), int(d['day_of_week']), season_enc,
                float(d['price']), float(d['discount_pct']), promo_enc,
                int(d['stock_level']), int(d['lead_time_days']),
                float(d['supplier_reliability']), float(d['historical_avg_demand']),
                float(d['demand_trend']), float(d['competitor_price_ratio']),
                float(d['customer_rating']), int(d['num_reviews']),
                int(d['reorder_point']), int(d['is_holiday_week']),
                float(d['region_demand_index']),
            ]], columns=self.features)
            pred = float(self.model.predict(fv)[0])
            results.append({
                'label': d.get('label', ''),
                'predicted_demand': round(pred)
            })
        return results

    def forecast_products(self, products, horizon_days=30, active_events=None):
        """
        Performs multi-product forecasting & inventory optimization for a list of product records.
        Applies seasonal & business events, calculates safety stock, ROP, recommended reorder, and statuses.
        """
        if not products:
            return []

        active_events = active_events or []
        today = datetime.date.today()
        month = today.month
        dow = today.weekday()

        # Determine season
        if month in [3, 4, 5]:
            season = 'Spring'
        elif month in [6, 7, 8]:
            season = 'Summer'
        elif month in [9, 10, 11]:
            season = 'Fall'
        else:
            season = 'Winter'

        feature_rows = []
        meta_items = []

        for p in products:
            cat = p['category']
            if cat not in self.encoders['product_category'].classes_:
                cat = 'Electronics' # Safe fallback

            cat_enc = self.encoders['product_category'].transform([cat])[0]
            season_enc = self.encoders['season'].transform([season])[0]

            # Check if any event applies to this category
            event_uplift = 0.0
            promo_type = 'None'
            for ev in active_events:
                if ev.get('category') in ['All', cat]:
                    event_uplift += float(ev.get('demand_uplift_pct', 0.0))
                    if ev.get('promotion_type') in self.encoders['promotion_type'].classes_:
                        promo_type = ev.get('promotion_type')

            if promo_type not in self.encoders['promotion_type'].classes_:
                promo_type = 'None'
            promo_enc = self.encoders['promotion_type'].transform([promo_type])[0]

            discount = 15.0 if promo_type != 'None' else 0.0
            is_holiday = 1 if len(active_events) > 0 else 0

            curr_stock = int(p['current_stock'])
            lead_time = int(p.get('lead_time_days') or 7)
            supplier_rel = float(p.get('supplier_reliability') or 0.88)
            hist_demand = float(p.get('historical_avg_demand') or 100.0)
            trend = float(p.get('demand_trend') or 0.0)
            comp_ratio = float(p.get('competitor_price_ratio') or 1.0)
            rating = float(p.get('customer_rating') or 4.5)
            reviews = int(p.get('num_reviews') or 150)
            rop = int(p.get('reorder_point') or 40)
            price = float(p['price'])
            cost = float(p.get('cost') or price * 0.5)

            row = [
                cat_enc, month, dow, season_enc,
                price, discount, promo_enc, curr_stock,
                lead_time, supplier_rel, hist_demand,
                trend, comp_ratio, rating, reviews, rop,
                is_holiday, 1.2 # region index
            ]
            feature_rows.append(row)
            meta_items.append({
                'product_id': p['product_id'],
                'name': p['name'],
                'category': cat,
                'price': price,
                'cost': cost,
                'current_stock': curr_stock,
                'lead_time_days': lead_time,
                'supplier_id': p.get('supplier_id', 'N/A'),
                'supplier_name': p.get('supplier_name', 'TechGlobal Logistics'),
                'supplier_reliability': supplier_rel,
                'historical_avg_demand': hist_demand,
                'event_uplift': event_uplift,
                'promotion_type': promo_type,
                'reorder_point_setting': rop
            })

        X = pd.DataFrame(feature_rows, columns=self.features)
        raw_monthly_preds = self.model.predict(X)

        # Confidence intervals across trees
        X_vals = X.values
        all_tree_preds = np.array([est.predict(X_vals) for est in self.model.estimators_])
        lower_cis = np.percentile(all_tree_preds, 10, axis=0)
        upper_cis = np.percentile(all_tree_preds, 90, axis=0)

        results = []
        horizon_factor = horizon_days / 30.0

        for i, meta in enumerate(meta_items):
            base_monthly = raw_monthly_preds[i]
            # Apply custom event uplift if active
            if meta['event_uplift'] > 0:
                base_monthly *= (1.0 + meta['event_uplift'] / 100.0)

            # Horizon adjusted demand
            forecast_demand = max(1, round(base_monthly * horizon_factor))
            lower_ci = max(1, round(lower_cis[i] * horizon_factor))
            upper_ci = max(lower_ci + 1, round(upper_cis[i] * horizon_factor))

            daily_demand = max(0.1, forecast_demand / float(horizon_days))
            ci_spread = max(1.0, upper_ci - lower_ci)
            std_dev_demand = ci_spread / 3.29 # approximation for 90-10 spread (approx 3.3 std devs)

            # Multi-factor Safety Stock incorporating supplier reliability & lead time
            # SS = Z * sqrt(L * sigma_d^2) * (1.5 - 0.5 * reliability)
            # Service level 95% => Z = 1.65
            supplier_risk_multiplier = 1.5 - (0.5 * meta['supplier_reliability'])
            safety_stock = round(1.65 * np.sqrt(meta['lead_time_days']) * (std_dev_demand / np.sqrt(horizon_days)) * supplier_risk_multiplier)
            safety_stock = max(5, safety_stock)

            # Reorder Point (ROP) = (Daily Demand * Lead Time) + Safety Stock
            reorder_point = round((daily_demand * meta['lead_time_days']) + safety_stock)

            curr_stock = meta['current_stock']
            # Recommended Order Quantity (Order up to target: Forecast Demand + Safety Stock - Current Stock)
            raw_rec_qty = (forecast_demand + safety_stock) - curr_stock
            recommended_order_qty = max(0, round(raw_rec_qty))

            # Status logic
            lead_time_demand = daily_demand * meta['lead_time_days']
            if curr_stock == 0 or curr_stock < max(3, round(lead_time_demand * 0.4)):
                status = 'Critical'
                reorder_needed = True
            elif curr_stock <= reorder_point:
                status = 'Low Stock'
                reorder_needed = True
            elif curr_stock > (forecast_demand * 2.0) and (curr_stock - forecast_demand > 40):
                status = 'Overstock'
                reorder_needed = False
            else:
                status = 'Normal'
                reorder_needed = False

            # Days of stock remaining
            days_of_stock = round(curr_stock / daily_demand, 1)

            # Confidence score (0 to 100)
            uncertainty_ratio = (upper_ci - lower_ci) / (2.0 * max(forecast_demand, 1))
            confidence_score = max(50, min(98, round(100 - (uncertainty_ratio * 30))))

            # Overstock excess & suggested actions
            excess_stock = max(0, curr_stock - (forecast_demand + safety_stock))
            excess_value = round(excess_stock * meta['cost'], 2)
            suggested_action = 'Maintain current purchase schedule'
            if status == 'Critical':
                suggested_action = f"URGENT: Issue emergency order of {recommended_order_qty} units with {meta['supplier_name']}"
            elif status == 'Low Stock':
                suggested_action = f"Standard Restock: Issue PO for {recommended_order_qty} units (Lead time: {meta['lead_time_days']} days)"
            elif status == 'Overstock':
                if excess_stock > 100:
                    suggested_action = "Promotional Clearance: 25% discount flash sale + pause future POs"
                elif excess_stock > 40:
                    suggested_action = "Cross-Sell Bundle: Package with high-velocity product + 15% discount"
                else:
                    suggested_action = "Slow Replenishment: Delay reorder date by 30 days"

            results.append({
                'product_id': meta['product_id'],
                'name': meta['name'],
                'category': meta['category'],
                'price': meta['price'],
                'cost': meta['cost'],
                'current_stock': curr_stock,
                'forecast_demand': forecast_demand,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'recommended_order_qty': recommended_order_qty,
                'status': status,
                'reorder_needed': reorder_needed,
                'days_of_stock': days_of_stock,
                'confidence_score': confidence_score,
                'lead_time_days': meta['lead_time_days'],
                'supplier_id': meta['supplier_id'],
                'supplier_name': meta['supplier_name'],
                'supplier_reliability': meta['supplier_reliability'],
                'historical_avg_demand': meta['historical_avg_demand'],
                'excess_stock': excess_stock,
                'excess_value': excess_value,
                'suggested_action': suggested_action,
                'inventory_value': round(curr_stock * meta['cost'], 2),
                'forecast_revenue': round(forecast_demand * meta['price'], 2)
            })

        return results

    def retrain_model_pipeline(self, sales_records, products_dict, min_r2=0.55):
        """
        Robust model retraining workflow:
        New historical data -> Data validation -> Preprocessing -> Train model ->
        Evaluate model -> Compare with current model -> Deploy only if configured criteria are met.
        """
        if len(sales_records) < 50:
            return {
                'success': False,
                'deployed': False,
                'message': f"Insufficient historical records ({len(sales_records)}). Retraining requires at least 50 validated records."
            }

        rows = []
        for s in sales_records:
            p_id = s.get('product_id')
            prod = products_dict.get(p_id, {})
            cat = prod.get('category', 'Electronics')
            if cat not in self.encoders['product_category'].classes_:
                cat = 'Electronics'

            try:
                sale_date = datetime.datetime.strptime(str(s['sale_date'])[:10], '%Y-%m-%d')
                month = sale_date.month
                dow = sale_date.weekday()
            except Exception:
                month = 6
                dow = 2

            if month in [3, 4, 5]:
                season = 'Spring'
            elif month in [6, 7, 8]:
                season = 'Summer'
            elif month in [9, 10, 11]:
                season = 'Fall'
            else:
                season = 'Winter'

            promo = s.get('promotion_type', 'None')
            if promo not in self.encoders['promotion_type'].classes_:
                promo = 'None'

            rows.append({
                'product_category_enc': self.encoders['product_category'].transform([cat])[0],
                'month': month,
                'day_of_week': dow,
                'season_enc': self.encoders['season'].transform([season])[0],
                'price': float(s.get('price', prod.get('price', 49.99))),
                'discount_pct': float(s.get('discount_pct', 0.0)),
                'promotion_type_enc': self.encoders['promotion_type'].transform([promo])[0],
                'stock_level': int(prod.get('current_stock', 100)),
                'lead_time_days': int(prod.get('lead_time_days', 7)),
                'supplier_reliability': float(prod.get('supplier_reliability', 0.88)),
                'historical_avg_demand': float(prod.get('historical_avg_demand', 120.0)),
                'demand_trend': float(prod.get('demand_trend', 0.05)),
                'competitor_price_ratio': float(prod.get('competitor_price_ratio', 1.0)),
                'customer_rating': float(prod.get('customer_rating', 4.5)),
                'num_reviews': int(prod.get('num_reviews', 250)),
                'reorder_point': int(prod.get('reorder_point', 40)),
                'is_holiday_week': 1 if s.get('is_holiday_week') or promo == 'Flash Sale' else 0,
                'region_demand_index': 1.2,
                'target_demand': float(s['sales_quantity'])
            })

        df = pd.DataFrame(rows)
        X = df[self.features]
        y = df['target_demand']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train candidate Random Forest model
        candidate_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
        candidate_model.fit(X_train, y_train)

        # Evaluate candidate model
        y_pred = candidate_model.predict(X_test)
        cand_mae = float(mean_absolute_error(y_test, y_pred))
        cand_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        cand_r2 = float(r2_score(y_test, y_pred))
        cand_mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100)

        current_metrics = self.metadata.get('metrics', {'mae': 102.17, 'rmse': 176.94, 'r2': 0.7307, 'mape': 45.47})
        
        # Validation Gate: Candidate must meet threshold
        is_acceptable = cand_r2 >= min_r2 or (cand_mae <= current_metrics['mae'] * 1.10)

        metrics_summary = {
            'candidate_metrics': {
                'mae': round(cand_mae, 2),
                'rmse': round(cand_rmse, 2),
                'r2': round(cand_r2, 4),
                'mape': round(cand_mape, 2)
            },
            'current_metrics': current_metrics,
            'samples_trained': len(X_train),
            'samples_evaluated': len(X_test)
        }

        if is_acceptable:
            # Backup current model
            try:
                joblib.dump(self.model, BACKUP_MODEL_PATH)
            except Exception:
                pass

            # Deploy candidate model
            joblib.dump(candidate_model, MODEL_PATH)
            self.model = candidate_model

            # Update metadata
            importances = pd.Series(candidate_model.feature_importances_, index=self.features).sort_values(ascending=False)
            self.metadata['metrics'] = metrics_summary['candidate_metrics']
            self.metadata['feature_importances'] = {k: round(float(v), 4) for k, v in importances.items()}
            self.metadata['train_size'] = len(X_train)
            self.metadata['test_size'] = len(X_test)
            self.metadata['last_retrained'] = datetime.datetime.now().isoformat()

            with open(METADATA_PATH, 'w') as f:
                json.dump(self.metadata, f, indent=2)

            return {
                'success': True,
                'deployed': True,
                'message': f"Model successfully retrained and deployed! R²: {cand_r2:.4f}, MAE: {cand_mae:.2f}",
                'metrics': metrics_summary
            }
        else:
            return {
                'success': False,
                'deployed': False,
                'message': f"Candidate model rejected by safety evaluation gate (R²: {cand_r2:.4f} < {min_r2:.2f}). Production model kept intact.",
                'metrics': metrics_summary
            }

# Global singleton
ml_engine = DemandMLEngine()
