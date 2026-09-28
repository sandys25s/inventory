"""
data_processor.py - Real Data Ingestion, Comprehensive Validation, and Sample Templates
Supports Excel (.xlsx) and CSV uploads with strict validation and friendly reporting.
"""
import os
import io
import datetime
import pandas as pd
import numpy as np
from database import get_db

REQUIRED_COLUMNS = {
    'product_id': ['product id', 'product_id', 'sku', 'item id', 'id'],
    'product_name': ['product name', 'product_name', 'name', 'item name', 'title'],
    'category': ['category', 'product category', 'department'],
    'date': ['date', 'sale date', 'transaction date', 'sale_date', 'timestamp'],
    'sales_quantity': ['sales quantity', 'sales_quantity', 'sales', 'quantity', 'units sold', 'demand'],
    'current_stock': ['current stock', 'current_stock', 'stock', 'inventory', 'stock level', 'on hand'],
    'price': ['price', 'unit price', 'selling price', 'retail price'],
    'discount': ['discount', 'discount %', 'discount_pct', 'discount rate'],
    'promotion': ['promotion', 'promotion type', 'promotion_type', 'promo'],
    'supplier': ['supplier', 'supplier name', 'vendor'],
    'supplier_reliability': ['supplier reliability', 'supplier_reliability', 'reliability', 'vendor score'],
    'lead_time': ['lead time', 'lead time days', 'lead_time_days', 'lead time (days)'],
    'region': ['region', 'territory', 'market', 'location']
}

STANDARD_COLUMNS = [
    'Product ID', 'Product Name', 'Category', 'Date', 'Sales Quantity',
    'Current Stock', 'Price', 'Discount', 'Promotion', 'Supplier',
    'Supplier Reliability', 'Lead Time', 'Region'
]

def map_column_names(df_columns):
    """Maps user-provided column headers to standard canonical keys."""
    mapping = {}
    normalized_cols = {str(c).strip().lower(): c for c in df_columns}
    
    for canon_key, aliases in REQUIRED_COLUMNS.items():
        found = None
        for alias in aliases:
            if alias in normalized_cols:
                found = normalized_cols[alias]
                break
        if found:
            mapping[found] = canon_key
            
    return mapping

def validate_and_process_file(file_storage, filename):
    """
    Parses Excel (.xlsx) or CSV file, performs strict business validation,
    cleanses records, and reports validation errors with row/column precision.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ['.csv', '.xlsx', '.xls']:
        return {
            'success': False,
            'errors': [f"Unsupported file format '{ext}'. Please upload a valid .csv or .xlsx file."],
            'summary': {}
        }

    try:
        if ext == '.csv':
            df = pd.read_csv(file_storage)
        else:
            df = pd.read_excel(file_storage)
    except Exception as e:
        return {
            'success': False,
            'errors': [f"Could not read the uploaded file: {str(e)}"],
            'summary': {}
        }

    if df.empty:
        return {
            'success': False,
            'errors': ["The uploaded file is empty (0 rows found)."],
            'summary': {}
        }

    # Map column names
    col_map = map_column_names(df.columns)
    missing_required = []
    
    essential_keys = ['product_id', 'product_name', 'category', 'date', 'sales_quantity', 'current_stock', 'price']
    for req in essential_keys:
        if req not in col_map.values():
            # Give helpful name
            readable_name = req.replace('_', ' ').title()
            missing_required.append(readable_name)

    if missing_required:
        return {
            'success': False,
            'errors': [
                f"Missing required columns in upload: {', '.join(missing_required)}.",
                f"Please ensure your header matches our template: {', '.join(STANDARD_COLUMNS)}"
            ],
            'summary': {'total_rows': len(df), 'valid_rows': 0, 'invalid_rows': len(df)}
        }

    # Rename to canonical internal columns
    df = df.rename(columns=col_map)

    # Fill optional columns with sensible defaults if missing
    if 'discount' not in df.columns:
        df['discount'] = 0.0
    if 'promotion' not in df.columns:
        df['promotion'] = 'None'
    if 'supplier' not in df.columns:
        df['supplier'] = 'Primary Supplier'
    if 'supplier_reliability' not in df.columns:
        df['supplier_reliability'] = 0.88
    if 'lead_time' not in df.columns:
        df['lead_time'] = 7
    if 'region' not in df.columns:
        df['region'] = 'North America'

    # Perform Row-by-Row Validation
    validation_errors = []
    cleaned_rows = []
    seen_keys = set()
    duplicate_count = 0

    valid_categories = ['Electronics', 'Clothing', 'Food & Beverage', 'Home & Garden', 'Sports', 'Toys', 'Books', 'Health & Beauty']
    valid_promos = ['None', 'Discount', 'Bundle', 'BOGO', 'Flash Sale']

    for idx, row in df.iterrows():
        row_num = idx + 2 # Excel 1-indexed, accounting for header
        row_errors = []

        # 1. Product ID
        p_id = str(row.get('product_id', '')).strip()
        if not p_id or p_id.lower() == 'nan':
            row_errors.append(f"Row {row_num}: Missing 'Product ID'")

        # 2. Product Name
        p_name = str(row.get('product_name', '')).strip()
        if not p_name or p_name.lower() == 'nan':
            row_errors.append(f"Row {row_num}: Missing 'Product Name'")

        # 3. Category
        cat = str(row.get('category', '')).strip()
        if not cat or cat.lower() == 'nan':
            cat = 'Electronics'
        else:
            matched_cat = next((c for c in valid_categories if c.lower() == cat.lower()), None)
            cat = matched_cat if matched_cat else 'Electronics'

        # 4. Date validation
        raw_date = row.get('date')
        parsed_date = None
        try:
            if pd.isna(raw_date):
                row_errors.append(f"Row {row_num}: Missing Date")
            else:
                parsed_date = pd.to_datetime(raw_date).strftime('%Y-%m-%d')
        except Exception:
            row_errors.append(f"Row {row_num}: Invalid Date format '{raw_date}'. Expected YYYY-MM-DD or MM/DD/YYYY.")

        # 5. Sales Quantity
        sales_qty = row.get('sales_quantity')
        try:
            if pd.isna(sales_qty):
                row_errors.append(f"Row {row_num}: Missing 'Sales Quantity'")
            else:
                sales_qty = float(sales_qty)
                if sales_qty < 0:
                    row_errors.append(f"Row {row_num}: Negative 'Sales Quantity' ({sales_qty}) is not permitted")
        except Exception:
            row_errors.append(f"Row {row_num}: Invalid numerical value for 'Sales Quantity'")

        # 6. Current Stock
        stock = row.get('current_stock')
        try:
            if pd.isna(stock):
                stock = 0
            else:
                stock = int(float(stock))
                if stock < 0:
                    row_errors.append(f"Row {row_num}: Current Stock cannot be negative ({stock})")
        except Exception:
            row_errors.append(f"Row {row_num}: Invalid numerical value for 'Current Stock'")

        # 7. Price
        price = row.get('price')
        try:
            if pd.isna(price):
                row_errors.append(f"Row {row_num}: Missing 'Price'")
            else:
                price = float(price)
                if price <= 0:
                    row_errors.append(f"Row {row_num}: Price must be greater than 0 ({price})")
        except Exception:
            row_errors.append(f"Row {row_num}: Invalid numerical value for 'Price'")

        # 8. Discount %
        discount = row.get('discount')
        try:
            discount = float(discount) if not pd.isna(discount) else 0.0
            if discount < 0 or discount > 100:
                row_errors.append(f"Row {row_num}: Discount must be between 0% and 100% ({discount})")
        except Exception:
            discount = 0.0

        # 9. Promotion
        promo = str(row.get('promotion', 'None')).strip()
        matched_promo = next((p for p in valid_promos if p.lower() == promo.lower()), 'None')

        # 10. Supplier & Reliability
        supplier = str(row.get('supplier', 'Primary Supplier')).strip()
        reliability = row.get('supplier_reliability')
        try:
            reliability = float(reliability) if not pd.isna(reliability) else 0.88
            # If specified as percentage (e.g. 88), convert to decimal 0.88
            if reliability > 1.0 and reliability <= 100.0:
                reliability = reliability / 100.0
            reliability = max(0.5, min(1.0, reliability))
        except Exception:
            reliability = 0.88

        # 11. Lead time
        lead_time = row.get('lead_time')
        try:
            lead_time = int(float(lead_time)) if not pd.isna(lead_time) else 7
            if lead_time < 1:
                lead_time = 7
        except Exception:
            lead_time = 7

        region = str(row.get('region', 'North America')).strip()

        # Check duplicates for same product and date
        dedup_key = (p_id, parsed_date)
        if dedup_key in seen_keys:
            duplicate_count += 1
            # Merge sales quantity into existing record or flag
            row_errors.append(f"Row {row_num}: Duplicate record found for Product '{p_id}' on Date '{parsed_date}'")
        else:
            seen_keys.add(dedup_key)

        if row_errors:
            validation_errors.extend(row_errors)
        else:
            cleaned_rows.append({
                'product_id': p_id,
                'name': p_name,
                'category': cat,
                'sale_date': parsed_date,
                'sales_quantity': sales_qty,
                'current_stock': stock,
                'price': price,
                'cost': round(price * 0.5, 2),
                'discount_pct': discount,
                'promotion_type': matched_promo,
                'supplier_name': supplier,
                'supplier_reliability': reliability,
                'lead_time_days': lead_time,
                'region': region
            })

    # If critical errors exceed threshold (e.g. more than 30 errors), cap displayed error list
    has_blocking_errors = len(validation_errors) > 0 and len(cleaned_rows) == 0

    return {
        'success': not has_blocking_errors,
        'has_warnings': len(validation_errors) > 0,
        'errors': validation_errors[:25],
        'total_errors': len(validation_errors),
        'duplicates_found': duplicate_count,
        'summary': {
            'total_rows': len(df),
            'valid_rows': len(cleaned_rows),
            'invalid_rows': len(df) - len(cleaned_rows)
        },
        'cleaned_rows': cleaned_rows
    }

def commit_uploaded_data(cleaned_rows, replace_existing=False):
    """
    Saves cleaned real business data into SQLite database.
    Marks records as is_demo = 0 (Real Business Data).
    """
    if not cleaned_rows:
        return 0, 0

    conn = get_db()
    cursor = conn.cursor()

    if replace_existing:
        # Clear previous real business data
        cursor.execute("DELETE FROM sales WHERE is_demo = 0")
        cursor.execute("DELETE FROM products WHERE is_demo = 0")

    # Group by product to update products table
    products_map = {}
    for r in cleaned_rows:
        pid = r['product_id']
        if pid not in products_map:
            products_map[pid] = {
                'product_id': pid,
                'name': r['name'],
                'category': r['category'],
                'price': r['price'],
                'cost': r['cost'],
                'current_stock': r['current_stock'],
                'reorder_point': max(20, round(r['sales_quantity'] * 0.3)),
                'lead_time_days': r['lead_time_days'],
                'supplier_id': f"SUP-{abs(hash(r['supplier_name'])) % 1000:03d}",
                'supplier_name': r['supplier_name'],
                'region': r['region'],
                'customer_rating': 4.5,
                'num_reviews': 120,
                'sales_quantities': [r['sales_quantity']],
                'supplier_reliability': r['supplier_reliability']
            }
        else:
            # Update stock to most recent
            products_map[pid]['current_stock'] = r['current_stock']
            products_map[pid]['sales_quantities'].append(r['sales_quantity'])

    # Ensure suppliers exist
    for p in products_map.values():
        cursor.execute('''
        INSERT OR IGNORE INTO suppliers (supplier_id, name, lead_time_days, reliability_score, on_time_delivery_pct, quality_rating)
        VALUES (?, ?, ?, ?, 92.0, 4.6)
        ''', (p['supplier_id'], p['supplier_name'], p['lead_time_days'], p['supplier_reliability']))

    # Insert or update products
    for p in products_map.values():
        avg_demand = float(np.mean(p['sales_quantities']))
        cursor.execute('''
        INSERT OR REPLACE INTO products (
            product_id, name, category, price, cost, current_stock,
            reorder_point, lead_time_days, supplier_id, region,
            customer_rating, num_reviews, historical_avg_demand,
            demand_trend, competitor_price_ratio, is_demo
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.05, 1.0, 0)
        ''', (
            p['product_id'], p['name'], p['category'], p['price'], p['cost'],
            p['current_stock'], p['reorder_point'], p['lead_time_days'],
            p['supplier_id'], p['region'], p['customer_rating'], p['num_reviews'],
            round(avg_demand, 1)
        ))

    # Insert sales records
    sales_to_insert = []
    for r in cleaned_rows:
        sales_to_insert.append((
            r['product_id'], r['sale_date'], r['sales_quantity'],
            r['price'], r['discount_pct'], r['promotion_type'],
            r['region'], 0 # is_demo = 0
        ))

    cursor.executemany('''
    INSERT INTO sales (product_id, sale_date, sales_quantity, price, discount_pct, promotion_type, region, is_demo)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', sales_to_insert)

    conn.commit()
    conn.close()

    return len(products_map), len(sales_to_insert)

def generate_sample_template(file_format='xlsx'):
    """
    Generates a realistic business inventory dataset as a downloadable sample template.
    Format: 'xlsx' or 'csv'.
    """
    today = datetime.date.today()
    sample_data = [
        # Product 1
        ['PRD-ENT-001', 'Smart Wireless Earbuds Pro', 'Electronics', (today - datetime.timedelta(days=60)).strftime('%Y-%m-%d'), 145, 85, 79.99, 10.0, 'Discount', 'TechGlobal Logistics', 0.92, 7, 'North America'],
        ['PRD-ENT-001', 'Smart Wireless Earbuds Pro', 'Electronics', (today - datetime.timedelta(days=30)).strftime('%Y-%m-%d'), 180, 85, 79.99, 15.0, 'Flash Sale', 'TechGlobal Logistics', 0.92, 7, 'North America'],
        ['PRD-ENT-001', 'Smart Wireless Earbuds Pro', 'Electronics', today.strftime('%Y-%m-%d'), 165, 85, 79.99, 0.0, 'None', 'TechGlobal Logistics', 0.92, 7, 'North America'],
        
        # Product 2 (Low stock)
        ['PRD-APP-002', 'Performance Dry-Fit T-Shirt', 'Clothing', (today - datetime.timedelta(days=60)).strftime('%Y-%m-%d'), 95, 12, 29.99, 0.0, 'None', 'Apex Apparel Supply', 0.85, 12, 'North America'],
        ['PRD-APP-002', 'Performance Dry-Fit T-Shirt', 'Clothing', (today - datetime.timedelta(days=30)).strftime('%Y-%m-%d'), 120, 12, 29.99, 10.0, 'Discount', 'Apex Apparel Supply', 0.85, 12, 'North America'],
        ['PRD-APP-002', 'Performance Dry-Fit T-Shirt', 'Clothing', today.strftime('%Y-%m-%d'), 110, 12, 29.99, 0.0, 'None', 'Apex Apparel Supply', 0.85, 12, 'North America'],

        # Product 3 (Overstock)
        ['PRD-HOM-003', 'Stainless Steel French Press', 'Home & Garden', (today - datetime.timedelta(days=60)).strftime('%Y-%m-%d'), 35, 260, 39.99, 0.0, 'None', 'Haven Home Solutions', 0.88, 14, 'North America'],
        ['PRD-HOM-003', 'Stainless Steel French Press', 'Home & Garden', (today - datetime.timedelta(days=30)).strftime('%Y-%m-%d'), 40, 260, 39.99, 0.0, 'None', 'Haven Home Solutions', 0.88, 14, 'North America'],
        ['PRD-HOM-003', 'Stainless Steel French Press', 'Home & Garden', today.strftime('%Y-%m-%d'), 38, 260, 39.99, 0.0, 'None', 'Haven Home Solutions', 0.88, 14, 'North America'],

        # Product 4
        ['PRD-FNB-004', 'Organic Cold Brew Blend 12oz', 'Food & Beverage', (today - datetime.timedelta(days=60)).strftime('%Y-%m-%d'), 210, 45, 18.50, 0.0, 'None', 'Evergreen Pantry Foods', 0.95, 4, 'North America'],
        ['PRD-FNB-004', 'Organic Cold Brew Blend 12oz', 'Food & Beverage', (today - datetime.timedelta(days=30)).strftime('%Y-%m-%d'), 245, 45, 18.50, 20.0, 'BOGO', 'Evergreen Pantry Foods', 0.95, 4, 'North America'],
        ['PRD-FNB-004', 'Organic Cold Brew Blend 12oz', 'Food & Beverage', today.strftime('%Y-%m-%d'), 230, 45, 18.50, 0.0, 'None', 'Evergreen Pantry Foods', 0.95, 4, 'North America'],

        # Product 5
        ['PRD-SPT-005', 'High-Speed Jump Rope Pro', 'Sports', (today - datetime.timedelta(days=60)).strftime('%Y-%m-%d'), 80, 95, 19.99, 0.0, 'None', 'Peak Athletics Distribution', 0.90, 8, 'North America'],
        ['PRD-SPT-005', 'High-Speed Jump Rope Pro', 'Sports', (today - datetime.timedelta(days=30)).strftime('%Y-%m-%d'), 95, 95, 19.99, 10.0, 'Bundle', 'Peak Athletics Distribution', 0.90, 8, 'North America'],
        ['PRD-SPT-005', 'High-Speed Jump Rope Pro', 'Sports', today.strftime('%Y-%m-%d'), 88, 95, 19.99, 0.0, 'None', 'Peak Athletics Distribution', 0.90, 8, 'North America']
    ]

    df = pd.DataFrame(sample_data, columns=STANDARD_COLUMNS)

    bio = io.BytesIO()
    if file_format.lower() == 'csv':
        df.to_csv(bio, index=False)
        mimetype = 'text/csv'
        filename = 'inventory_data_template.csv'
    else:
        with pd.ExcelWriter(bio, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Inventory_Sales_Data')
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        filename = 'inventory_data_template.xlsx'

    bio.seek(0)
    return bio, mimetype, filename
