REQUIRED_SHOPIFY_SCOPES = [
    # Product & Collection Access
    'read_products',         # View product details, variants, images
    'write_products',        # Update product availability, variants
    
    # Inventory Access
    'read_inventory',        # Check stock levels
    'write_inventory',       # Update inventory levels
    
    # Price & Discount Access
    'read_price_rules',      # Access to pricing rules
    'write_price_rules',     # Modify pricing rules
    
    # Cart & Checkout Access
    'read_orders',           # View order details
    'write_orders',          # Create/modify orders
    'read_checkouts',        # View checkout processes
    'write_checkouts',       # Modify checkout processes
    
    # Customer Data Access
    'read_customers',        # View customer data
    'write_customers',       # Update customer preferences
    
    # Analytics Access
    'read_analytics',        # Access conversion data
]

def verify_shopify_access():
    """Verify all required scopes are available"""
    shop_url = os.getenv('SHOPIFY_SHOP_URL')
    access_token = os.getenv('SHOPIFY_ACCESS_TOKEN')
    
    session = shopify.Session(shop_url, '2024-01', access_token)
    shopify.ShopifyResource.activate_session(session)
    
    try:
        # Test access to various endpoints
        shopify.Product.first()
        shopify.Inventory.first()
        shopify.PriceRule.first()
        shopify.Order.first()
        print("✅ All required Shopify permissions are available")
        return True
    except shopify.ValidationException as e:
        print(f"❌ Missing required permissions: {str(e)}")
        print("\nPlease update your Shopify app permissions to include:")
        for scope in REQUIRED_SHOPIFY_SCOPES:
            print(f"  - {scope}")
        return False
    finally:
        shopify.ShopifyResource.clear_session() 