from collections import defaultdict
import shopify
from typing import Dict, List
from datetime import datetime, timedelta

class ShopifyAPIAgent:
    def __init__(self, shop_url: str, access_token: str):
        self.shop_url = shop_url
        self.access_token = access_token
        self._setup_shopify()
        
    def _setup_shopify(self):
        """Initialize Shopify session"""
        session = shopify.Session(self.shop_url, '2024-01', self.access_token)
        shopify.ShopifyResource.activate_session(session)
        
    async def analyze_checkouts(self) -> Dict:
        """Comprehensive checkout analysis using real store data"""
        print("\n🔍 Analyzing Store Data...")
        
        try:
            # Get last 30 days of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # Fetch data
            checkouts = shopify.Checkout.find(
                created_at_min=start_date.isoformat(),
                created_at_max=end_date.isoformat(),
                limit=250
            )
            
            orders = shopify.Order.find(
                created_at_min=start_date.isoformat(),
                created_at_max=end_date.isoformat(),
                limit=250
            )
            
            # Analyze checkout data
            checkout_data = self._analyze_checkouts(checkouts)
            order_data = self._analyze_orders(orders)
            
            return {
                'period': '30_days',
                'checkouts': checkout_data,
                'orders': order_data,
                'conversion_rate': self._calculate_conversion_rate(checkout_data, order_data),
                'completion_rate': self._calculate_completion_rate(checkout_data),
                'revenue_metrics': self._calculate_revenue_metrics(order_data)
            }
            
        except Exception as e:
            print(f"Error fetching store data: {str(e)}")
            return self._generate_empty_results()
    
    def _analyze_checkouts(self, checkouts: List) -> Dict:
        """Detailed checkout analysis"""
        analysis = {
            'total_count': 0,
            'abandoned_count': 0,
            'device_breakdown': defaultdict(int),
            'payment_methods': defaultdict(int),
            'shipping_rates': defaultdict(int),
            'customer_types': defaultdict(int),
            'time_to_abandon': defaultdict(int),
            'steps_completed': defaultdict(int)
        }
        
        for checkout in checkouts:
            analysis['total_count'] += 1
            
            # Track device type
            analysis['device_breakdown'][checkout.get('device_id', 'unknown')] += 1
            
            # Track customer type
            customer_type = 'returning' if checkout.get('customer') else 'guest'
            analysis['customer_types'][customer_type] += 1
            
            if not checkout.completed_at:
                analysis['abandoned_count'] += 1
                # Track abandonment step
                analysis['steps_completed'][checkout.get('current_step', 'unknown')] += 1
        
        return analysis
    
    def _analyze_orders(self, orders: List) -> Dict:
        """Detailed order analysis"""
        analysis = {
            'total_count': 0,
            'total_revenue': 0,
            'avg_order_value': 0,
            'payment_methods': defaultdict(int),
            'customer_segments': defaultdict(int),
            'products_per_order': defaultdict(int),
            'discount_usage': defaultdict(int)
        }
        
        for order in orders:
            analysis['total_count'] += 1
            analysis['total_revenue'] += float(order.total_price)
            
            # Track payment method
            analysis['payment_methods'][order.get('gateway', 'unknown')] += 1
            
            # Track products per order
            product_count = len(order.line_items)
            analysis['products_per_order'][product_count] += 1
            
            # Track discount usage
            if order.discount_codes:
                analysis['discount_usage']['used'] += 1
        
        if analysis['total_count'] > 0:
            analysis['avg_order_value'] = analysis['total_revenue'] / analysis['total_count']
        
        return analysis
    
    def _calculate_conversion_rate(self, checkout_data: Dict, order_data: Dict) -> float:
        """Calculate store conversion rate"""
        if checkout_data['total_count'] == 0:
            return 0
        return (order_data['total_count'] / checkout_data['total_count']) * 100
    
    def _calculate_completion_rate(self, checkout_data: Dict) -> float:
        """Calculate checkout completion rate"""
        if checkout_data['total_count'] == 0:
            return 0
        completed = checkout_data['total_count'] - checkout_data['abandoned_count']
        return (completed / checkout_data['total_count']) * 100
    
    def _calculate_revenue_metrics(self, order_data: Dict) -> Dict:
        """Calculate detailed revenue metrics"""
        return {
            'total_revenue': order_data['total_revenue'],
            'avg_order_value': order_data['avg_order_value'],
            'revenue_per_customer': order_data['total_revenue'] / order_data['total_count'] if order_data['total_count'] > 0 else 0,
            'discount_impact': len(order_data['discount_usage']) / order_data['total_count'] if order_data['total_count'] > 0 else 0
        }
    
    def _generate_empty_results(self) -> Dict:
        """Generate empty results structure"""
        return {
            'period': '30_days',
            'checkouts': {
                'total_count': 0,
                'abandoned_count': 0,
                'device_breakdown': {},
                'customer_types': {}
            },
            'orders': {
                'total_count': 0,
                'total_revenue': 0,
                'avg_order_value': 0
            },
            'conversion_rate': 0,
            'completion_rate': 0,
            'revenue_metrics': {
                'total_revenue': 0,
                'avg_order_value': 0,
                'revenue_per_customer': 0
            }
        }
