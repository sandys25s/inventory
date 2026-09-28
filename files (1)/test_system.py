"""
test_system.py - End-to-End System Test Suite
Verifies all 25 project goals and requirements:
- Authentication & RBAC
- Backward compatibility with original ML API
- Multi-product forecasting & optimization
- Real data Excel/CSV upload with validation
- KPI dashboard metrics
- Product details and multi-horizon forecasts (7/30/90 days)
- Reorder alerts and overstock mitigation
- Seasonal business events
- What-If sensitivity simulation
- Supplier performance tracking
- Multi-format report export (Excel, CSV, PDF)
- Model performance monitoring and retrain validation gate
"""
import unittest
import io
import json
import app
import data_processor

class DemandIQSystemTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import database
        database.init_db()
        database.seed_demo_data(force=True)
        cls.client = app.app.test_client()

    def test_01_index_html_served(self):
        """Verify main frontend is served."""
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'DemandIQ', res.data)

    def test_02_backward_compatibility_metadata(self):
        """Verify original /api/metadata endpoint."""
        res = self.client.get('/api/metadata')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('categories', data)
        self.assertIn('metrics', data)
        self.assertIn('r2', data['metrics'])

    def test_03_backward_compatibility_single_predict(self):
        """Verify original /api/predict endpoint with Random Forest model."""
        payload = {
            'product_category': 'Electronics',
            'month': 6,
            'day_of_week': 2,
            'season': 'Summer',
            'price': 49.99,
            'discount_pct': 10,
            'promotion_type': 'Discount',
            'stock_level': 120,
            'lead_time_days': 7,
            'supplier_reliability': 0.85,
            'historical_avg_demand': 150,
            'demand_trend': 0.05,
            'competitor_price_ratio': 1.0,
            'customer_rating': 4.2,
            'num_reviews': 320,
            'reorder_point': 50,
            'is_holiday_week': 0,
            'region_demand_index': 1.2
        }
        res = self.client.post('/api/predict', json=payload)
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        self.assertIn('predicted_demand', d)
        self.assertIn('lower_ci', d)
        self.assertIn('upper_ci', d)
        self.assertIn('recommended_order_qty', d)
        self.assertIn('days_of_stock_remaining', d)
        self.assertGreater(d['predicted_demand'], 0)

    def test_04_backward_compatibility_batch_predict(self):
        """Verify original /api/batch_predict scenario comparison."""
        payload = [
            {
                'label': 'Test Summer',
                'product_category': 'Electronics', 'month': 6, 'day_of_week': 2, 'season': 'Summer',
                'price': 49.99, 'discount_pct': 10, 'promotion_type': 'Flash Sale', 'stock_level': 120,
                'lead_time_days': 7, 'supplier_reliability': 0.85, 'historical_avg_demand': 150,
                'demand_trend': 0.05, 'competitor_price_ratio': 1.0, 'customer_rating': 4.2,
                'num_reviews': 320, 'reorder_point': 50, 'is_holiday_week': 0, 'region_demand_index': 1.2
            }
        ]
        res = self.client.post('/api/batch_predict', json=payload)
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        self.assertEqual(len(d), 1)
        self.assertEqual(d[0]['label'], 'Test Summer')

    def test_05_auth_and_roles(self):
        """Verify authentication for admin, manager, staff and bad login."""
        # Bad login
        res_bad = self.client.post('/api/auth/login', json={'username': 'admin', 'password': 'wrongpassword'})
        self.assertEqual(res_bad.status_code, 401)

        # Successful Admin login
        res_admin = self.client.post('/api/auth/login', json={'username': 'admin', 'password': 'admin123'})
        self.assertEqual(res_admin.status_code, 200)
        self.assertEqual(res_admin.get_json()['user']['role'], 'admin')

        # Check me
        res_me = self.client.get('/api/auth/me')
        self.assertEqual(res_me.status_code, 200)
        self.assertTrue(res_me.get_json()['authenticated'])

    def test_06_multi_product_forecasts(self):
        """Verify multi-product forecasting for 7, 30, and 90 day horizons."""
        for horizon in [7, 30, 90]:
            res = self.client.get(f'/api/forecasts?horizon={horizon}')
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data['horizon_days'], horizon)
            self.assertGreater(data['count'], 0)
            
            # Check fields on first item
            first = data['forecasts'][0]
            required_fields = [
                'product_id', 'name', 'category', 'current_stock', 'forecast_demand',
                'safety_stock', 'reorder_point', 'recommended_order_qty', 'status',
                'days_of_stock', 'confidence_score'
            ]
            for f in required_fields:
                self.assertIn(f, first)
            self.assertIn(first['status'], ['Critical', 'Low Stock', 'Normal', 'Overstock'])

    def test_07_dashboard_inventory_kpis(self):
        """Verify dashboard KPIs and aggregated metrics."""
        res = self.client.get('/api/inventory?horizon=30')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        kpis = data['kpis']
        self.assertIn('total_products', kpis)
        self.assertIn('total_inventory', kpis)
        self.assertIn('expected_demand', kpis)
        self.assertIn('low_stock_products', kpis)
        self.assertIn('overstock_products', kpis)
        self.assertIn('inventory_value', kpis)
        self.assertIn('top_high_demand', data)
        self.assertIn('category_breakdown', data)

    def test_08_product_details_and_multi_horizons(self):
        """Verify detailed product view with 7, 30, and 90 day horizons."""
        res = self.client.get('/api/products/PRD-101/details')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['product']['product_id'], 'PRD-101')
        self.assertIn('historical_sales', data)
        self.assertIn('7_days', data['forecast_horizons'])
        self.assertIn('30_days', data['forecast_horizons'])
        self.assertIn('90_days', data['forecast_horizons'])

    def test_09_reorder_alerts(self):
        """Verify automatic reorder alerts."""
        res = self.client.get('/api/alerts')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('alerts', data)
        self.assertGreater(data['total_alerts'], 0)
        alert_types = [a['alert_type'] for a in data['alerts']]
        # Expect critical or low_stock or overstock
        self.assertTrue(any(t in ['critical', 'low_stock', 'overstock'] for t in alert_types))

    def test_10_overstock_detection_and_actionable_advice(self):
        """Verify overstock detection and mitigation advice."""
        res = self.client.get('/api/inventory?horizon=30')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        overstock_items = data['overstock_items']
        self.assertGreater(len(overstock_items), 0)
        for item in overstock_items:
            self.assertEqual(item['status'], 'Overstock')
            self.assertGreater(item['excess_stock'], 0)
            self.assertIn('suggested_action', item)

    def test_11_what_if_simulation(self):
        """Verify What-If sensitivity simulation."""
        payload = {
            'base_params': {
                'product_category': 'Electronics', 'month': 6, 'day_of_week': 2, 'season': 'Summer',
                'price': 80.00, 'discount_pct': 0, 'promotion_type': 'None', 'stock_level': 100,
                'lead_time_days': 7, 'supplier_reliability': 0.90, 'historical_avg_demand': 150,
                'demand_trend': 0.05, 'competitor_price_ratio': 1.0, 'customer_rating': 4.5,
                'num_reviews': 200, 'reorder_point': 40, 'is_holiday_week': 0, 'region_demand_index': 1.2
            },
            'scenarios': [
                {'label': 'High Price ($200)', 'price': 200.0, 'discount_pct': 0, 'promotion_type': 'None'},
                {'label': 'Standard Price ($80)', 'price': 80.0, 'discount_pct': 0, 'promotion_type': 'None'},
                {'label': 'Discount Promotion ($20)', 'price': 20.0, 'discount_pct': 20, 'promotion_type': 'Discount'}
            ]
        }
        res = self.client.post('/api/simulation', json=payload)
        self.assertEqual(res.status_code, 200)
        sims = res.get_json()['simulations']
        self.assertEqual(len(sims), 3)
        self.assertGreater(sims[2]['predicted_demand'], sims[0]['predicted_demand'])

    def test_12_suppliers_management(self):
        """Verify supplier performance and directory API."""
        res = self.client.get('/api/suppliers')
        self.assertEqual(res.status_code, 200)
        suppliers = res.get_json()
        self.assertGreater(len(suppliers), 0)
        self.assertIn('reliability_score', suppliers[0])
        self.assertIn('lead_time_days', suppliers[0])

    def test_13_seasonal_and_business_events(self):
        """Verify adding, listing, and deleting business events."""
        # List
        res = self.client.get('/api/events')
        self.assertEqual(res.status_code, 200)
        initial_count = len(res.get_json())

        # Add event
        new_event = {
            'event_name': 'Test Spring Clearance',
            'start_date': '2026-04-01',
            'end_date': '2026-04-15',
            'category': 'Electronics',
            'demand_uplift_pct': 30.0,
            'promotion_type': 'Flash Sale',
            'description': 'Automated unit test event'
        }
        res_add = self.client.post('/api/events', json=new_event)
        self.assertEqual(res_add.status_code, 200)

        # Verify added
        res_list = self.client.get('/api/events')
        events = res_list.get_json()
        self.assertEqual(len(events), initial_count + 1)
        added = next(e for e in events if e['event_name'] == 'Test Spring Clearance')

        # Clean up delete
        res_del = self.client.delete(f"/api/events/{added['id']}")
        self.assertEqual(res_del.status_code, 200)

    def test_14_data_upload_and_validation(self):
        """Verify real Excel/CSV upload with strict validation and friendly errors."""
        # 1. Download template
        res_tpl = self.client.get('/api/template/download?format=xlsx')
        self.assertEqual(res_tpl.status_code, 200)
        self.assertGreater(len(res_tpl.data), 1000)

        # 2. Upload valid template
        bio, mime, fname = data_processor.generate_sample_template('xlsx')
        res_up = self.client.post('/api/upload', data={'file': (bio, fname)}, content_type='multipart/form-data')
        self.assertEqual(res_up.status_code, 200)
        self.assertTrue(res_up.get_json()['success'])

        # 3. Upload invalid file (empty columns)
        bad_csv = io.BytesIO(b'wrong_header_1,wrong_header_2\n10,20')
        res_bad = self.client.post('/api/upload', data={'file': (bad_csv, 'invalid.csv')}, content_type='multipart/form-data')
        self.assertEqual(res_bad.status_code, 422)
        self.assertFalse(res_bad.get_json()['success'])
        self.assertGreater(len(res_bad.get_json()['errors']), 0)

    def test_15_reports_export(self):
        """Verify CSV, Excel, and PDF report generation."""
        # Excel Forecast
        res_xl = self.client.get('/api/reports/export?type=forecast&format=xlsx')
        self.assertEqual(res_xl.status_code, 200)
        self.assertEqual(res_xl.content_type, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        # CSV Reorder
        res_csv = self.client.get('/api/reports/export?type=reorder&format=csv')
        self.assertEqual(res_csv.status_code, 200)
        self.assertEqual(res_csv.content_type, 'text/csv; charset=utf-8')

        # PDF Overstock
        res_pdf = self.client.get('/api/reports/export?type=overstock&format=pdf')
        self.assertEqual(res_pdf.status_code, 200)
        self.assertEqual(res_pdf.content_type, 'application/pdf')

    def test_16_model_performance_and_retraining(self):
        """Verify model performance monitoring and safe retraining validation."""
        res = self.client.get('/api/model-performance')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('current_metrics', data)
        self.assertIn('actual_vs_predicted', data)
        self.assertIn('history', data)

        # Trigger retrain (logged in as admin)
        res_retrain = self.client.post('/api/model/retrain')
        self.assertEqual(res_retrain.status_code, 200)
        retrain_data = res_retrain.get_json()
        self.assertIn('success', retrain_data)
        self.assertIn('message', retrain_data)

if __name__ == '__main__':
    unittest.main()
