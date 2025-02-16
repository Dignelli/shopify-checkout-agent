import requests
from typing import Dict, List

class CartScorer:
    def __init__(self):
        self.ACCESS_TOKEN = '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'
        self.STORE_URL = 'cartscore2.myshopify.com'
        self.headers = {
            'X-Shopify-Access-Token': self.ACCESS_TOKEN,
            'Content-Type': 'application/json'
        }

    def get_cart_data(self, cart_id: str) -> Dict:
        """Get data for a specific cart"""
        url = f'https://{self.STORE_URL}/admin/api/2024-01/carts/{cart_id}.json'
        response = requests.get(url, headers=self.headers)
        return response.json()

    def calculate_cart_score(self, cart_data: Dict) -> float:
        """Calculate a score for the cart based on various factors"""
        score = 0.0
        
        # Cart Value Score (0-40 points)
        cart_total = float(cart_data.get('total_price', 0))
        if cart_total > 100:
            score += 40
        elif cart_total > 50:
            score += 25
        elif cart_total > 20:
            score += 10
        
        # Item Count Score (0-30 points)
        item_count = len(cart_data.get('line_items', []))
        if item_count >= 3:
            score += 30
        elif item_count >= 2:
            score += 20
        elif item_count >= 1:
            score += 10
            
        # Customer Status (0-30 points)
        if cart_data.get('customer'):
            score += 30
        
        return score

    def get_active_carts(self) -> List[Dict]:
        """Get all active carts"""
        # Try different endpoints to find the cart data
        endpoints = [
            '/admin/api/2024-01/checkouts.json',
            '/admin/api/2024-01/draft_orders.json',
            '/admin/api/2024-01/orders.json?status=any'
        ]

        for endpoint in endpoints:
            url = f'https://{self.STORE_URL}{endpoint}'
            print(f"Trying endpoint: {url}")  # Debug output
            
            response = requests.get(url, headers=self.headers)
            print(f"Response status: {response.status_code}")  # Debug output
            print(f"Response body: {response.text[:200]}...")  # Show first 200 chars
            
            if response.status_code == 200:
                data = response.json()
                return data.get('checkouts', []) or data.get('draft_orders', []) or data.get('orders', [])
        
        return []

if __name__ == "__main__":
    scorer = CartScorer()
    carts = scorer.get_active_carts()
    print(f"\nFound {len(carts)} active carts")
    
    for cart in carts:
        print(f"\nCart details: {cart}")
    
    # Test scoring each cart
    for cart in carts:
        score = scorer.calculate_cart_score(cart)
        print(f"Cart {cart.get('id')}: Score = {score}") 