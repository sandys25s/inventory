"""
database.py - SQLite Database Layer for Inventory Demand Forecasting & Optimization
Supports Users, Suppliers, Products, Sales, Forecasts, Alerts, Orders, Events, and Model Performance.
"""
import sqlite3
import os
import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inventory.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes database schema if tables do not exist."""
    conn = get_db()
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL, -- 'admin', 'manager', 'staff'
        full_name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Suppliers table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS suppliers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        supplier_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        contact_email TEXT,
        phone TEXT,
        lead_time_days INTEGER DEFAULT 7,
        reliability_score REAL DEFAULT 0.85, -- 0.0 to 1.0
        on_time_delivery_pct REAL DEFAULT 90.0,
        quality_rating REAL DEFAULT 4.5,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Products table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL,
        cost REAL NOT NULL,
        current_stock INTEGER DEFAULT 0,
        reorder_point INTEGER DEFAULT 50,
        lead_time_days INTEGER DEFAULT 7,
        supplier_id TEXT,
        region TEXT DEFAULT 'North America',
        customer_rating REAL DEFAULT 4.2,
        num_reviews INTEGER DEFAULT 100,
        historical_avg_demand REAL DEFAULT 120,
        demand_trend REAL DEFAULT 0.05,
        competitor_price_ratio REAL DEFAULT 1.0,
        is_demo INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
    )
    ''')

    # Sales history table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id TEXT NOT NULL,
        sale_date TEXT NOT NULL,
        sales_quantity REAL NOT NULL,
        price REAL NOT NULL,
        discount_pct REAL DEFAULT 0.0,
        promotion_type TEXT DEFAULT 'None',
        region TEXT DEFAULT 'North America',
        is_demo INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    )
    ''')

    # Forecasts table (caches recent product-level predictions)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id TEXT NOT NULL,
        forecast_date TEXT NOT NULL,
        period_days INTEGER DEFAULT 30,
        predicted_demand REAL NOT NULL,
        lower_ci REAL NOT NULL,
        upper_ci REAL NOT NULL,
        safety_stock REAL NOT NULL,
        reorder_point REAL NOT NULL,
        recommended_order_qty REAL NOT NULL,
        status TEXT NOT NULL, -- 'Critical', 'Low Stock', 'Normal', 'Overstock'
        reorder_needed INTEGER DEFAULT 0,
        days_of_stock REAL DEFAULT 30.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    )
    ''')

    # Alerts table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id TEXT NOT NULL,
        alert_type TEXT NOT NULL, -- 'critical', 'low_stock', 'overstock', 'surge_demand'
        priority TEXT NOT NULL, -- 'high', 'medium', 'low'
        message TEXT NOT NULL,
        current_stock INTEGER NOT NULL,
        forecast_demand REAL NOT NULL,
        recommended_action TEXT NOT NULL,
        resolved INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    )
    ''')

    # Purchase Orders table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_number TEXT UNIQUE NOT NULL,
        supplier_id TEXT NOT NULL,
        product_id TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        unit_cost REAL NOT NULL,
        total_cost REAL NOT NULL,
        status TEXT DEFAULT 'pending', -- 'pending', 'shipped', 'delivered', 'cancelled'
        order_date TEXT NOT NULL,
        expected_delivery TEXT,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Business Events table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS business_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_name TEXT NOT NULL,
        start_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        category TEXT DEFAULT 'All',
        demand_uplift_pct REAL DEFAULT 20.0,
        promotion_type TEXT DEFAULT 'Discount',
        description TEXT,
        is_active INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Model Performance tracking table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        evaluation_date TEXT NOT NULL,
        model_version TEXT NOT NULL,
        mae REAL NOT NULL,
        rmse REAL NOT NULL,
        r2 REAL NOT NULL,
        mape REAL NOT NULL,
        dataset_type TEXT DEFAULT 'demo_data', -- 'demo_data' or 'real_business_data'
        samples_count INTEGER NOT NULL,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    conn.commit()
    conn.close()

    # Seed initial accounts if users table is empty
    seed_users()

def seed_users():
    """Seeds default users for role-based access control."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) as cnt FROM users')
    if cursor.fetchone()['cnt'] == 0:
        default_users = [
            ('admin', generate_password_hash('admin123'), 'admin', 'System Administrator'),
            ('manager', generate_password_hash('manager123'), 'manager', 'Inventory Manager'),
            ('staff', generate_password_hash('staff123'), 'staff', 'Operations Staff')
        ]
        cursor.executemany(
            'INSERT INTO users (username, password_hash, role, full_name) VALUES (?, ?, ?, ?)',
            default_users
        )
        conn.commit()
    conn.close()

def seed_demo_data(force=False):
    """Populates realistic demo data for products, suppliers, sales, and events."""
    init_db()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) as cnt FROM products WHERE is_demo = 1')
    existing_cnt = cursor.fetchone()['cnt']
    if existing_cnt > 0 and not force:
        conn.close()
        return

    if force:
        cursor.execute('DELETE FROM sales WHERE is_demo = 1')
        cursor.execute('DELETE FROM forecasts WHERE product_id IN (SELECT product_id FROM products WHERE is_demo = 1)')
        cursor.execute('DELETE FROM alerts WHERE product_id IN (SELECT product_id FROM products WHERE is_demo = 1)')
        cursor.execute('DELETE FROM products WHERE is_demo = 1')
        cursor.execute('DELETE FROM suppliers')
        cursor.execute('DELETE FROM business_events')

    # 1. Seed Suppliers
    suppliers_data = [
        ('SUP-001', 'TechGlobal Logistics', 'contact@techglobal.io', '+1-800-555-0101', 7, 0.94, 96.5, 4.8, 'Tier-1 Electronics & Peripherals Supplier'),
        ('SUP-002', 'Apex Apparel Supply', 'orders@apexapparel.com', '+1-800-555-0102', 12, 0.88, 91.0, 4.4, 'Primary Garment Manufacturer'),
        ('SUP-003', 'Evergreen Pantry Foods', 'supply@evergreenpantry.com', '+1-800-555-0103', 4, 0.96, 97.2, 4.9, 'Fresh & Packaged Food Supplier'),
        ('SUP-004', 'Haven Home Solutions', 'sales@havenhome.com', '+1-800-555-0104', 14, 0.82, 85.0, 4.2, 'Furniture & Kitchen Appliances'),
        ('SUP-005', 'Peak Athletics Distribution', 'b2b@peakathletic.com', '+1-800-555-0105', 9, 0.91, 93.4, 4.6, 'Sports Equipment & Activewear'),
        ('SUP-006', 'WonderPlay Toys Co', 'info@wonderplay.com', '+1-800-555-0106', 15, 0.79, 82.5, 4.1, 'Educational Toys & Board Games'),
        ('SUP-007', 'Veda Health & Wellness', 'orders@vedahealth.com', '+1-800-555-0107', 6, 0.93, 95.0, 4.7, 'Cosmetics & Nutritional Supplements'),
        ('SUP-008', 'Meridian Media & Publishing', 'dist@meridianpress.com', '+1-800-555-0108', 5, 0.95, 98.0, 4.8, 'Technical Books & Paper Goods')
    ]
    cursor.executemany('''
    INSERT OR REPLACE INTO suppliers (supplier_id, name, contact_email, phone, lead_time_days, reliability_score, on_time_delivery_pct, quality_rating, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', suppliers_data)

    # 2. Seed 24 Realistic Products across 8 Categories
    products_data = [
        # Electronics
        ('PRD-101', 'Pro Wireless Mouse', 'Electronics', 59.99, 28.50, 18, 45, 7, 'SUP-001', 'North America', 4.7, 850, 160, 0.12, 0.95, 1),
        ('PRD-102', '4K Ultra USB-C Monitor 27"', 'Electronics', 349.99, 195.00, 12, 25, 10, 'SUP-001', 'North America', 4.8, 1240, 85, 0.08, 0.98, 1),
        ('PRD-103', 'Mechanical RGB Gaming Keyboard', 'Electronics', 89.99, 42.00, 110, 40, 8, 'SUP-001', 'North America', 4.6, 620, 120, 0.05, 1.02, 1),
        ('PRD-104', 'Noise-Cancelling BT Headphones', 'Electronics', 149.99, 72.00, 310, 50, 7, 'SUP-001', 'North America', 4.9, 2150, 110, -0.02, 0.92, 1), # Overstock

        # Clothing
        ('PRD-201', 'Men Classic Oxford Shirt', 'Clothing', 45.00, 18.00, 8, 35, 12, 'SUP-002', 'North America', 4.3, 310, 95, 0.04, 1.00, 1), # Low stock
        ('PRD-202', 'Performance Athletic Hoodie', 'Clothing', 65.00, 26.00, 75, 50, 12, 'SUP-002', 'North America', 4.5, 480, 130, 0.15, 0.96, 1),
        ('PRD-203', 'Slim-Fit Stretch Denim Jeans', 'Clothing', 55.00, 22.00, 140, 60, 12, 'SUP-002', 'North America', 4.4, 520, 115, 0.02, 1.05, 1),
        ('PRD-204', 'Thermal Winter Parka Jacket', 'Clothing', 180.00, 85.00, 280, 30, 15, 'SUP-002', 'North America', 4.7, 790, 60, -0.10, 1.10, 1), # Overstock

        # Food & Beverage
        ('PRD-301', 'Artisan Dark Roast Coffee (1kg)', 'Food & Beverage', 24.99, 11.20, 5, 40, 4, 'SUP-003', 'North America', 4.8, 1420, 210, 0.18, 0.90, 1), # Critical stock
        ('PRD-302', 'Organic Japanese Matcha (100g)', 'Food & Beverage', 29.99, 13.50, 65, 30, 5, 'SUP-003', 'North America', 4.7, 610, 80, 0.06, 0.98, 1),
        ('PRD-303', 'Plant Protein Energy Bars (12pk)', 'Food & Beverage', 27.50, 12.00, 190, 70, 4, 'SUP-003', 'North America', 4.4, 380, 160, 0.09, 1.00, 1),

        # Home & Garden
        ('PRD-401', 'Ergonomic Lumbar Office Chair', 'Home & Garden', 229.00, 115.00, 14, 20, 14, 'SUP-004', 'North America', 4.6, 940, 70, 0.11, 0.95, 1), # Low stock
        ('PRD-402', 'Smart LED Floor Lamp', 'Home & Garden', 79.99, 36.00, 85, 35, 12, 'SUP-004', 'North America', 4.3, 420, 65, 0.03, 1.04, 1),
        ('PRD-403', 'Cast Iron 5-Piece Cookware Set', 'Home & Garden', 129.99, 58.00, 220, 25, 16, 'SUP-004', 'North America', 4.8, 1120, 45, -0.05, 1.08, 1), # Overstock

        # Sports
        ('PRD-501', 'Eco-Grip High Density Yoga Mat', 'Sports', 38.00, 15.50, 95, 40, 9, 'SUP-005', 'North America', 4.7, 730, 125, 0.08, 0.97, 1),
        ('PRD-502', 'Adjustable Dumbbells Set 50lb', 'Sports', 249.00, 120.00, 16, 25, 10, 'SUP-005', 'North America', 4.8, 1650, 80, 0.14, 0.92, 1), # Low stock
        ('PRD-503', 'Hydration Pack Running Vest', 'Sports', 69.99, 31.00, 82, 30, 8, 'SUP-005', 'North America', 4.5, 390, 90, 0.06, 1.01, 1),

        # Toys
        ('PRD-601', 'Programmable STEM Robot Kit', 'Toys', 89.99, 39.00, 45, 30, 15, 'SUP-006', 'North America', 4.6, 510, 75, 0.07, 0.98, 1),
        ('PRD-602', 'Magnetic Architectural Tiles 100pc', 'Toys', 49.99, 21.00, 160, 50, 14, 'SUP-006', 'North America', 4.8, 890, 110, 0.10, 0.95, 1),
        ('PRD-603', 'Classic Wooden Train Play Set', 'Toys', 59.99, 25.00, 240, 35, 16, 'SUP-006', 'North America', 4.5, 420, 50, -0.08, 1.05, 1), # Overstock

        # Health & Beauty
        ('PRD-701', 'Pure Vitamin C Facial Radiance Serum', 'Health & Beauty', 34.00, 11.50, 120, 50, 6, 'SUP-007', 'North America', 4.8, 1850, 175, 0.16, 0.94, 1),
        ('PRD-702', 'Daily Mineral Sunscreen SPF 50', 'Health & Beauty', 26.50, 9.80, 210, 60, 6, 'SUP-007', 'North America', 4.7, 1340, 190, 0.22, 0.92, 1),

        # Books
        ('PRD-801', 'Production ML System Architecture', 'Books', 49.99, 18.00, 65, 30, 5, 'SUP-008', 'North America', 4.9, 780, 85, 0.10, 0.90, 1),
        ('PRD-802', 'Supply Chain Analytics & Optimization', 'Books', 54.99, 20.00, 80, 25, 5, 'SUP-008', 'North America', 4.7, 430, 60, 0.05, 0.96, 1)
    ]

    cursor.executemany('''
    INSERT OR REPLACE INTO products (
        product_id, name, category, price, cost, current_stock, reorder_point,
        lead_time_days, supplier_id, region, customer_rating, num_reviews,
        historical_avg_demand, demand_trend, competitor_price_ratio, is_demo
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', products_data)

    # 3. Seed 12 Months of Historical Sales Data
    today = datetime.date.today()
    sales_records = []
    
    # Generate 12 months of monthly aggregated sales records for each product
    for p in products_data:
        p_id = p[0]
        base_demand = p[12]
        trend = p[13]
        price = p[3]
        cat = p[2]

        for m_back in range(12, 0, -1):
            sale_date = (today - datetime.timedelta(days=m_back * 30)).strftime('%Y-%m-%d')
            # Add seasonal variation
            month_num = (today.month - m_back) % 12 + 1
            season_factor = 1.25 if month_num in [11, 12, 6, 7] else 0.95
            if cat == 'Food & Beverage':
                season_factor *= 1.15
            elif cat == 'Clothing' and month_num in [10, 11, 12]:
                season_factor *= 1.3

            qty = round(max(5, base_demand * (1 + trend * (12 - m_back) / 6.0) * season_factor))
            promo = 'Flash Sale' if month_num in [11, 7] else ('Discount' if month_num % 3 == 0 else 'None')
            discount = 15.0 if promo != 'None' else 0.0

            sales_records.append((
                p_id, sale_date, qty, price, discount, promo, 'North America', 1
            ))

    cursor.executemany('''
    INSERT INTO sales (product_id, sale_date, sales_quantity, price, discount_pct, promotion_type, region, is_demo)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', sales_records)

    # 4. Seed Business Events
    events = [
        ('Diwali Festival Mega Sale', f"{today.year}-10-20", f"{today.year}-11-05", 'All', 35.0, 'Flash Sale', 'Annual festive shopping surge across all major categories.', 1),
        ('Black Friday & Cyber Week', f"{today.year}-11-22", f"{today.year}-11-30", 'Electronics', 45.0, 'Discount', 'High demand promotional period for electronics and home appliances.', 1),
        ('Summer Sports & Fitness Kickoff', f"{today.year}-06-01", f"{today.year}-06-25", 'Sports', 25.0, 'Bundle', 'Peak seasonal demand for fitness, apparel, and outdoor goods.', 1),
        ('Back to School & University Reopening', f"{today.year}-08-15", f"{today.year}-09-10", 'Books', 40.0, 'Discount', 'Surge in academic, technical books, stationery, and computing supplies.', 1),
        ('New Year Clearance & Reset', f"{today.year}-01-02", f"{today.year}-01-18", 'Health & Beauty', 30.0, 'BOGO', 'Post-holiday health resolutions and skincare promotions.', 1)
    ]
    cursor.executemany('''
    INSERT OR REPLACE INTO business_events (event_name, start_date, end_date, category, demand_uplift_pct, promotion_type, description, is_active)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', events)

    # 5. Seed initial model performance record
    cursor.execute('''
    INSERT INTO model_performance (evaluation_date, model_version, mae, rmse, r2, mape, dataset_type, samples_count, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        today.strftime('%Y-%m-%d'), 'RF-v1.0-Baseline', 102.17, 176.94, 0.7307, 45.47, 'demo_data', 5000,
        'Initial baseline trained on synthetic cross-category demand distribution.'
    ))

    conn.commit()
    conn.close()

if __name__ == '__main__':
    print("Initializing Database...")
    init_db()
    seed_demo_data(force=True)
    print("Database successfully initialized and seeded with demo data!")
