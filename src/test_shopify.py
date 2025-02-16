import os
import shopify
from dotenv import load_dotenv

def test_connection():
    # Load environment variables from .env file
    load_dotenv('src/.env')
    
    # Get credentials from environment
    shop_url = os.getenv('SHOPIFY_SHOP_URL')
    access_token = os.getenv('SHOPIFY_ACCESS_TOKEN')
    
    if not shop_url or not access_token:
        print("❌ Missing environment variables:")
        if not shop_url:
            print("  - SHOPIFY_SHOP_URL")
        if not access_token:
            print("  - SHOPIFY_ACCESS_TOKEN")
        return False
    
    print(f"Testing connection to: {shop_url}")
    
    try:
        session = shopify.Session(shop_url, '2024-01', access_token)
        shopify.ShopifyResource.activate_session(session)
        
        # Get shop details
        shop = shopify.Shop.current()
        print(f"\n✅ Successfully connected to {shop.name}")
        
        # Get products
        products = shopify.Product.find()
        print("\nProducts in store:")
        for product in products:
            print(f"- {product.title}")
            print(f"  Price: ${product.variants[0].price}")
            print(f"  Inventory: {product.variants[0].inventory_quantity}")
        
        # Get recent orders
        orders = shopify.Order.find(limit=5)
        print("\nRecent orders:")
        for order in orders:
            print(f"- Order #{order.order_number}")
            print(f"  Total: ${order.total_price}")
            print(f"  Status: {order.financial_status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False
    finally:
        shopify.ShopifyResource.clear_session()

if __name__ == "__main__":
    test_connection() 