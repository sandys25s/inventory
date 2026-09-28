# DemandIQ™ — Enterprise Inventory Forecasting & Optimization System

DemandIQ is a production-style SaaS application for Machine Learning-powered Inventory Demand Forecasting, Safety Stock Planning, Overstock Mitigation, and Supply Chain Optimization.

The system builds upon the original Random Forest demand prediction model and expands it into an enterprise platform with real business data ingestion, multi-product forecasting, automated restock alerts, interactive What-If sensitivity simulations, supplier performance tracking, multi-format reporting, and a validated model retraining pipeline.

---

## 🌟 Key Features

1. **Real Data Upload (Excel & CSV)**
   - Upload real business inventory datasets (`.xlsx`, `.xls`, `.csv`).
   - Supports required business columns: *Product ID, Product Name, Category, Date, Sales Quantity, Current Stock, Price, Discount, Promotion, Supplier, Supplier Reliability, Lead Time, Region*.
   - Strict row-by-row data validation reporting missing fields, negative stock/sales, invalid dates, and duplicates.
   - Built-in sample Excel & CSV template generators.

2. **Multi-Product Forecasting & Optimization**
   - High-throughput batch forecasting across all catalog items.
   - Multi-horizon projection selector: **7 Days**, **30 Days**, and **90 Days**.
   - Calculates **Safety Stock** ($SS = Z \times \sqrt{L \cdot \sigma_D^2} \times \text{Supplier Risk Factor}$).
   - Computes **Reorder Point (ROP)** and **Recommended Reorder Quantity**.
   - Categorizes SKU health into statuses: `Critical`, `Low Stock`, `Normal`, and `Overstock`.

3. **Executive Dashboard**
   - 7 High-level KPI cards: *Total Products, Total Inventory, Expected Demand, Low Stock, Overstock, Products to Reorder, Inventory Valuation ($)*.
   - Interactive charts via Chart.js:
     - Historical Actual Sales vs ML Forecast
     - 12-Month Demand Trend & Seasonality Index
     - Category-wise Demand Breakdown
     - Top 10 High Demand Products
     - Supplier Performance Radar

4. **Product Details View**
   - Deep-dive into any individual SKU.
   - 12-month historical sales timeline vs ML forecast trajectory.
   - Multi-horizon forecast comparisons (7d, 30d, 90d).
   - Supplier reliability, lead time, and reorder metrics.

5. **Autonomous Reorder Alerts**
   - Prioritized alerts (`High Priority Reorder`, `Low Stock Warning`, `Demand Surge`).
   - Real-time days-of-stock coverage estimates.

6. **Overstock Detection & Working Capital Recovery**
   - Detects inventory significantly exceeding demand.
   - Computes trapped working capital ($).
   - Generates actionable business recovery recommendations: promotional clearance discounts (15-25%), cross-sell product bundles, and purchase order postponements.

7. **Configurable Seasonal & Business Events**
   - Configure holidays, festivals (e.g. Diwali, Black Friday, Christmas, Back to School, Summer Sale), and promotional campaigns.
   - Custom category uplift multipliers (+20% to +50%) and promotional types without hardcoded assumptions.

8. **What-If Sensitivity Simulation**
   - Dynamic simulation of Price, Discount %, Promotion Type, Lead Time, and Stock changes.
   - Side-by-side scenario comparison table and dual-axis revenue/demand sensitivity chart.

9. **Supplier Reliability Management**
   - Comprehensive supplier directory tracking lead times, reliability scores (%), on-time delivery rates, and quality ratings.
   - Safety stock and ROP calculations incorporate supplier performance.

10. **Multi-Format Report Export**
    - Instant generation of **Excel (.xlsx)**, **CSV**, and **PDF** reports for:
      - Demand Forecast & Inventory Optimization
      - Restock & Reorder Action Plans
      - Overstock Working Capital Recovery
      - Supplier Reliability Audits
      - ML Model Performance Audits

11. **ML Model Governance & Retraining**
    - Tracks accuracy metrics: **MAE**, **RMSE**, **R² Score**, and **MAPE**.
    - Feature importance analysis.
    - Actual vs Predicted tracking table.
    - Admin-gated retraining pipeline: trains candidate model, evaluates against holdout data, and deploys only if validation criteria (e.g. $R^2 \ge 0.55$) are satisfied. Preserves backup model.

12. **Role-Based Access Control (RBAC)**
    - Secure password hashing via `werkzeug.security`.
    - Three roles: **Admin**, **Inventory Manager**, and **Staff**.

13. **Demo Data Mode vs Real Business Data Mode**
    - Clear demarcation between synthetic demo data and real uploaded business data.
    - One-click "Reload Demo Data" button for sandbox testing.

14. **Full Backward Compatibility**
    - Preserves all original endpoints: `/api/metadata`, `/api/predict`, and `/api/batch_predict`.
    - The original single prediction interface and scenario comparison tool remain fully functional.

---

## 🗄️ Database Schema (`inventory.db`)

- `users`: User credentials, password hashes, and roles (`admin`, `manager`, `staff`).
- `suppliers`: Supplier profiles, lead times, reliability scores, on-time delivery percentages.
- `products`: SKU master, category, price, cost, stock, ROP, lead time, supplier FK, demo flag.
- `sales`: Historical transaction records, sales quantities, dates, promotions, discounts.
- `forecasts`: Cached predictions, safety stocks, ROPs, recommended reorders, statuses.
- `alerts`: Operational stockout and overstock alerts with recommended actions.
- `orders`: Purchase orders, quantities, costs, delivery tracking.
- `business_events`: Configured seasonal events, start/end dates, demand uplifts, promo types.
- `model_performance`: Audit trail of model versions, MAE, RMSE, R², and dataset sources.

---

## 🚀 Getting Started

### 1. Requirements
- Python 3.10+
- Installed packages:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Initialize Database & Seed Demo Data
```bash
python database.py
```

### 3. Run the Application
```bash
python app.py
```
Open your web browser and navigate to:
```
http://localhost:5050/
```

### 4. Run Automated Test Suite
To verify all 16 end-to-end integration tests:
```bash
python test_system.py
```

---

## 🔑 Test Credentials

| Role | Username | Password | Permitted Actions |
|---|---|---|---|
| **System Admin** | `admin` | `admin123` | Full access: User management, model retraining, event creation, uploads |
| **Inventory Manager** | `manager` | `manager123` | Forecasts, alerts, stock adjustment, reports, supplier viewing |
| **Operations Staff** | `staff` | `staff123` | View catalog, forecast queries, single prediction engine |

*A 1-click credential selector is built into the login modal.*

---

## 📡 API Endpoints Reference

### Authentication & Users
- `POST /api/auth/login`: Authenticate user and initialize session
- `POST /api/auth/logout`: Destroy session
- `GET /api/auth/me`: Get current authenticated user profile
- `GET /api/users`: List users (Admin)
- `POST /api/users`: Create user (Admin)

### Original & Core ML
- `GET /api/metadata`: Get model features, categories, and metrics (Preserved)
- `POST /api/predict`: Predict demand for single parameter vector (Preserved)
- `POST /api/batch_predict`: Predict demand for multiple scenarios (Preserved)

### Forecasting & Inventory
- `GET /api/forecasts?horizon=30&category=All`: Multi-product forecasts with safety stock and ROP
- `GET /api/inventory?horizon=30`: Dashboard KPI metrics and category breakdowns
- `PUT /api/inventory/stock`: Adjust stock quantity for an SKU
- `GET /api/products`: List product catalog with search and filters
- `POST /api/products`: Add new SKU
- `GET /api/products/<product_id>/details`: Detailed product view (7d/30d/90d projections)

### Alerts & Optimization
- `GET /api/alerts`: Operational alerts (Critical, Low Stock, Overstock, Surge)
- `POST /api/simulation`: What-If parameter sensitivity analysis

### Suppliers & Events
- `GET /api/suppliers`: Supplier directory and performance metrics
- `POST /api/suppliers`: Add supplier profile
- `GET /api/events`: List seasonal and promotional business events
- `POST /api/events`: Configure new event with category demand uplift
- `DELETE /api/events/<id>`: Remove event

### Uploads & Reports
- `GET /api/template/download?format=xlsx|csv`: Download sample inventory template
- `POST /api/upload`: Upload real business data with row-by-row validation
- `GET /api/reports/export?type=forecast|reorder|overstock|suppliers|performance&format=xlsx|csv|pdf`: Export reports
- `GET /api/model-performance`: View live accuracy metrics and actual vs predicted log
- `POST /api/model/retrain`: Retrain model on verified sales data (Admin)
- `POST /api/demo/load`: Reset/reload demo dataset

---

## 📋 Sample Upload Format (Excel / CSV)

| Product ID | Product Name | Category | Date | Sales Quantity | Current Stock | Price | Discount | Promotion | Supplier | Supplier Reliability | Lead Time | Region |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PRD-ENT-001 | Smart Wireless Earbuds Pro | Electronics | 2026-08-15 | 145 | 85 | 79.99 | 10.0 | Discount | TechGlobal Logistics | 0.92 | 7 | North America |
| PRD-APP-002 | Performance Dry-Fit T-Shirt | Clothing | 2026-08-15 | 95 | 12 | 29.99 | 0.0 | None | Apex Apparel Supply | 0.85 | 12 | North America |
| PRD-HOM-003 | Stainless Steel French Press | Home & Garden | 2026-08-15 | 35 | 260 | 39.99 | 0.0 | None | Haven Home Solutions | 0.88 | 14 | North America |

---

## 🔮 Future Improvements
1. Integration with external ERP/WMS APIs (SAP, Oracle NetSuite, Shopify, WooCommerce).
2. Deep learning / Transformer-based multi-horizon time-series architectures (e.g. Temporal Fusion Transformers, DeepAR) alongside Random Forest.
3. Automated purchase order PDF generation and direct EDI transmission to suppliers.
4. Multi-location / multi-warehouse stock balancing and transfer recommendations.
