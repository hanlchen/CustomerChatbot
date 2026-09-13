"""
Record access for the support tools.

Two backends behind one set of functions. If `customer.db` exists, customers
and orders are read from SQLite; otherwise they are generated in memory from a
fixed seed. The eleven public functions keep the same signatures and return
the same dicts either way, so nothing above this layer knows which is in use.

The seed is fixed so CUST-10001 is the same person on every restart and in
every process. The global RNG state is saved and restored around generation,
so seeding the data does not make the rest of the app's randomness
predictable.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import random
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class OrderStatus(Enum):
    """Order status values."""
    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURNED = "returned"


class ShippingStatus(Enum):
    """Shipping status values."""
    NOT_SHIPPED = "not_shipped"
    IN_TRANSIT = "in_transit"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    EXCEPTION = "exception"



# ============================================================================
# Database Implementation
# ============================================================================

# Fixed seed for the generated dataset. Change it to get a different (but
# still stable) set of demo records.
DATA_SEED = 20260817


class CustomerChatbotDatabase:
    """In-memory database for customer support data."""

    def __init__(self):
        self.customers: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.products: Dict[str, Dict[str, Any]] = {}
        self.policies: List[Dict[str, Any]] = []
        self.faqs: List[Dict[str, Any]] = []
        self.support_tickets: Dict[str, Dict[str, Any]] = {}
        self._initialize_data()

    def _initialize_data(self):
        """Initialize the database with production data.

        Generation is seeded so the dataset is identical in every process and
        after every restart. Without this, CUST-10001 was a different person
        each run -- orders, emails and return windows all changed underneath
        the customer, and any ID shared between the app and a test was
        meaningless.

        The global RNG state is saved and restored so seeding here doesn't
        make the rest of the application's randomness predictable.
        """
        previous_state = random.getstate()
        random.seed(DATA_SEED)
        try:
            self._load_customers()
            self._load_products()
            self._load_orders()
            self._load_policies()
            self._load_faqs()
            self._load_support_tickets()
        finally:
            random.setstate(previous_state)

    def _load_customers(self):
        """Load customer data (50+ realistic customers)."""
        first_names = [
            "John", "Jane", "Michael", "Emily", "David", "Sarah", "James", "Jessica",
            "Robert", "Linda", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
            "Thomas", "Karen", "Christopher", "Nancy", "Daniel", "Lisa", "Matthew", "Betty",
            "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley", "Steven", "Kimberly",
            "Paul", "Donna", "Andrew", "Carol", "Joshua", "Michelle", "Kenneth", "Amanda",
            "Kevin", "Deborah", "Brian", "Stephanie", "George", "Rebecca", "Edward", "Sharon",
            "Ronald", "Laura", "Timothy", "Cynthia"
        ]

        last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
            "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
            "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
            "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
            "Young", "Allen", "King", "Wright", "Scott", "Torres", "Peterson", "Phillips",
            "Campbell", "Parker", "Evans", "Edwards", "Collins", "Reeves", "Stewart", "Morris"
        ]

        cities = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
            "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
            "Fort Worth", "Columbus", "Charlotte", "San Francisco", "Indianapolis", "Seattle",
            "Denver", "Boston", "Atlanta", "Nashville", "Portland", "Las Vegas", "Detroit"
        ]

        states = ["NY", "CA", "IL", "TX", "AZ", "PA", "TX", "CA", "TX", "CA", "TX", "FL",
                  "TX", "OH", "NC", "CA", "IN", "WA", "CO", "MA", "GA", "TN", "OR", "NV", "MI"]

        shipping_methods = ["standard", "express", "overnight", "ground"]
        tiers = ["bronze", "silver", "gold", "platinum"]

        base_date = datetime.now() - timedelta(days=365)

        for i in range(55):
            customer_id = f"CUST-{10000 + i}"
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            city_idx = i % len(cities)

            customer = {
                "customer_id": customer_id,
                "first_name": first_name,
                "last_name": last_name,
                "email": f"{first_name.lower()}.{last_name.lower()}{i}@email.com",
                "phone": f"+1{random.randint(200, 999)}{random.randint(200, 999)}{random.randint(1000, 9999)}",
                "billing_address": {
                    "street_line1": f"{random.randint(1, 9999)} {random.choice(['Main', 'Oak', 'Elm', 'Park', 'Market', 'Spring'])} St",
                    "street_line2": None if random.random() > 0.3 else f"Suite {random.randint(100, 999)}",
                    "city": cities[city_idx],
                    "state_province": states[city_idx],
                    "postal_code": f"{random.randint(10000, 99999)}",
                    "country": "USA"
                },
                "shipping_address": {
                    "street_line1": f"{random.randint(1, 9999)} {random.choice(['Main', 'Oak', 'Elm', 'Park', 'Market', 'Spring'])} St",
                    "street_line2": None if random.random() > 0.3 else f"Suite {random.randint(100, 999)}",
                    "city": cities[(city_idx + random.randint(-2, 2)) % len(cities)],
                    "state_province": states[(city_idx + random.randint(-2, 2)) % len(states)],
                    "postal_code": f"{random.randint(10000, 99999)}",
                    "country": "USA"
                },
                "preferred_shipping_method": random.choice(shipping_methods),
                "created_at": (base_date + timedelta(days=random.randint(0, 365))).isoformat(),
                "tier": random.choice(tiers),
                "lifetime_value": round(random.uniform(50, 5000), 2),
                "total_orders": random.randint(0, 20),
                "last_order_date": (datetime.now() - timedelta(days=random.randint(0, 90))).isoformat() if random.random() > 0.2 else None
            }
            self.customers[customer_id] = customer

    def _load_products(self):
        """Load product catalog (30+ realistic products)."""
        products_data = [
            ("Wireless Headphones", "Electronics", "Premium noise-cancelling wireless headphones with 30hr battery life", 79.99, 35.00, 150),
            ("USB-C Cable 6ft", "Electronics", "High-quality USB-C charging and data cable", 12.99, 2.50, 500),
            ("Phone Screen Protector", "Accessories", "Tempered glass screen protector for most smartphones", 9.99, 1.20, 300),
            ("Laptop Stand", "Office", "Adjustable aluminum laptop stand for better posture", 34.99, 12.00, 80),
            ("Wireless Mouse", "Electronics", "Ergonomic wireless mouse with 2.4GHz connectivity", 24.99, 8.00, 200),
            ("Mechanical Keyboard", "Electronics", "RGB mechanical gaming keyboard with Cherry MX switches", 129.99, 45.00, 75),
            ("Phone Case", "Accessories", "Protective TPU phone case with shock absorption", 19.99, 4.50, 400),
            ("Portable Charger", "Electronics", "20000mAh portable power bank with dual USB ports", 34.99, 12.00, 120),
            ("HDMI Cable 10ft", "Electronics", "4K HDMI 2.0 cable with gold-plated connectors", 14.99, 2.80, 250),
            ("Desk Lamp", "Office", "LED desk lamp with adjustable brightness and color temperature", 49.99, 18.00, 90),
            ("Webcam 1080p", "Electronics", "1080p HD webcam with built-in microphone", 59.99, 20.00, 110),
            ("Monitor Stand", "Office", "Monitor riser with storage drawer underneath", 39.99, 14.00, 100),
            ("Keyboard Wrist Rest", "Office", "Memory foam wrist rest for keyboard comfort", 16.99, 5.00, 180),
            ("Mouse Pad", "Office", "Large extended mouse pad with non-slip base", 22.99, 6.00, 250),
            ("USB Hub 4-Port", "Electronics", "Compact 4-port USB 3.0 hub with power adapter", 29.99, 9.00, 140),
            ("Phone Stand", "Accessories", "Adjustable phone stand for desk and table", 13.99, 3.50, 320),
            ("Cable Organizer Set", "Office", "5-piece cable management organizer kit", 14.99, 4.00, 280),
            ("Backup Drive 2TB", "Electronics", "2TB external hard drive for data backup", 99.99, 35.00, 60),
            ("SD Card 128GB", "Electronics", "High-speed 128GB microSD card for cameras", 44.99, 15.00, 95),
            ("Cooling Pad", "Electronics", "Laptop cooling pad with dual fans", 24.99, 8.50, 170),
            ("Screen Cleaner Kit", "Accessories", "Microfiber cloth and cleaner spray for screens", 8.99, 1.50, 400),
            ("Monitor Light Bar", "Office", "USB-powered monitor light bar for glare reduction", 79.99, 28.00, 85),
            ("Desk Chair Mat", "Office", "Heavy-duty chair mat for hardwood floors", 44.99, 15.00, 110),
            ("Document Holder", "Office", "Adjustable document holder for typing reference", 19.99, 6.00, 200),
            ("Pen Set Premium", "Office", "Set of 12 premium ballpoint pens", 24.99, 7.00, 150),
            ("Notebook A4", "Office", "Premium lined notebook 200 pages", 12.99, 3.50, 500),
            ("Desk Organizer", "Office", "Multi-compartment desk organizer with drawers", 34.99, 11.00, 130),
            ("Time Lock Safe", "Accessories", "Programmable time-lock safe box", 89.99, 30.00, 45),
            ("Power Strip 6-Outlet", "Electronics", "Surge-protected 6-outlet power strip with USB ports", 22.99, 7.00, 220),
            ("Microphone USB", "Electronics", "Cardioid USB microphone for streaming and recording", 69.99, 24.00, 100),
        ]

        for i, (name, category, description, price, cost, stock) in enumerate(products_data):
            product_id = f"PROD-{1000 + i}"
            self.products[product_id] = {
                "id": product_id,
                "name": name,
                "category": category,
                "description": description,
                "price": price,
                "cost": cost,
                "stock_level": stock,
                "return_window_days": random.choice([14, 30, 60]),
                "warranty_months": random.choice([1, 6, 12, 24]),
                "rating": round(random.uniform(3.5, 5.0), 1),
                "review_count": random.randint(5, 500),
                "sku": f"SKU-{10000 + i}"
            }

    def _load_orders(self):
        """Load order data (200+ realistic orders)."""
        base_date = datetime.now() - timedelta(days=180)
        tracking_prefixes = ["1Z", "9400", "9200", "CP", "LH"]

        for order_num in range(250):
            order_id = f"ORD-{100000 + order_num}"
            customer_id = random.choice(list(self.customers.keys()))
            customer = self.customers[customer_id]

            # Create 1-5 items per order
            num_items = random.randint(1, 5)
            items = []
            subtotal = 0

            for _ in range(num_items):
                product_id = random.choice(list(self.products.keys()))
                product = self.products[product_id]
                quantity = random.randint(1, 3)
                unit_price = product["price"]
                total_price = quantity * unit_price

                items.append({
                    "product_id": product_id,
                    "product_name": product["name"],
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "total_price": round(total_price, 2),
                    "category": product["category"],
                    "return_eligible": True
                })
                subtotal += total_price

            order_date = base_date + timedelta(days=random.randint(0, 180))
            status = random.choice([
                OrderStatus.DELIVERED.value,
                OrderStatus.DELIVERED.value,
                OrderStatus.DELIVERED.value,
                OrderStatus.SHIPPED.value,
                OrderStatus.PROCESSING.value,
                OrderStatus.CANCELLED.value,
                OrderStatus.RETURNED.value,
            ])

            # Determine shipping and delivery status
            if status == OrderStatus.DELIVERED.value:
                shipping_status = ShippingStatus.DELIVERED.value
                estimated_delivery = order_date + timedelta(days=random.randint(3, 7))
                actual_delivery = estimated_delivery + timedelta(days=random.randint(-2, 2))
            elif status == OrderStatus.SHIPPED.value:
                shipping_status = random.choice([
                    ShippingStatus.IN_TRANSIT.value,
                    ShippingStatus.OUT_FOR_DELIVERY.value
                ])
                estimated_delivery = datetime.now() + timedelta(days=random.randint(1, 5))
                actual_delivery = None
            else:
                shipping_status = ShippingStatus.NOT_SHIPPED.value
                estimated_delivery = order_date + timedelta(days=random.randint(3, 7))
                actual_delivery = None

            tax = round(subtotal * 0.08, 2)
            shipping_cost = round(random.uniform(5.99, 24.99), 2)
            total_amount = round(subtotal + tax + shipping_cost, 2)

            self.orders[order_id] = {
                "order_id": order_id,
                "customer_id": customer_id,
                "order_date": order_date.isoformat(),
                "status": status,
                "items": items,
                "subtotal": round(subtotal, 2),
                "tax": tax,
                "shipping_cost": shipping_cost,
                "total_amount": total_amount,
                "shipping_address": customer["shipping_address"],
                "tracking_number": f"{random.choice(tracking_prefixes)}{random.randint(100000000, 999999999)}" if status in [OrderStatus.SHIPPED.value, OrderStatus.DELIVERED.value] else None,
                "estimated_delivery": estimated_delivery.isoformat() if estimated_delivery else None,
                "actual_delivery": actual_delivery.isoformat() if actual_delivery else None,
                "return_eligible": status == OrderStatus.DELIVERED.value and (datetime.now() - order_date).days < 30,
                "return_deadline": (order_date + timedelta(days=30)).date().isoformat() if status == OrderStatus.DELIVERED.value else None,
                "shipping_status": shipping_status,
                "notes": random.choice([None, "Left at front door", "Signature required", "Will call held at facility"]) if status == OrderStatus.DELIVERED.value else None
            }

    def _load_policies(self):
        """Load the company policy documents."""
        policies_data = [
            {
                "id": "RET-001",
                "title": "30-Day Return Policy",
                "category": "Returns",
                "content": """Our 30-Day Return Policy allows you to return most items purchased from our store for a full refund within 30 days of purchase.

Eligibility:
- Items must be in original condition with all packaging and accessories
- Clothing must have tags attached and show no signs of wear
- Electronics must be in working condition
- Items purchased on clearance are non-returnable

Process:
1. Contact our support team with your order number
2. We'll provide a prepaid shipping label
3. Ship the item back to our warehouse
4. Once received and inspected, your refund will be processed within 5-7 business days

Exceptions:
- Final sale items (marked at checkout)
- Custom or personalized items
- Items damaged due to misuse""",
                "last_updated": "2025-06-15",
                "version": "3.0",
                "applicable_regions": ["USA", "Canada"]
            },
            {
                "id": "SHP-001",
                "title": "Shipping Policy and Costs",
                "category": "Shipping",
                "content": """We offer multiple shipping options to meet your needs.

Shipping Methods:
1. Standard Shipping (5-7 business days): FREE on orders over $50, $5.99 otherwise
2. Express Shipping (2-3 business days): $12.99
3. Overnight Shipping (1 business day): $24.99
4. Ground Shipping (3-5 business days): $3.99

Regional Rates:
- Continental USA: Standard rates apply
- Alaska/Hawaii: Add $10 to any method
- Canada: Add $5 to any method
- International: Calculated at checkout

Processing Time:
Orders are processed and shipped within 1-2 business days of purchase. Orders placed on weekends or holidays ship the next business day.

Tracking:
All orders include tracking information sent via email.""",
                "last_updated": "2025-07-01",
                "version": "2.5",
                "applicable_regions": ["USA", "Canada", "International"]
            },
            {
                "id": "WAR-001",
                "title": "Warranty Coverage",
                "category": "Warranty",
                "content": """Product warranties vary by category.

Electronics (12-month):
Covers defects in materials and workmanship. Does not cover accidental damage, misuse, or normal wear.

Accessories (6-month):
Coverage includes manufacturing defects and material failures.

Office Products (6-month):
Covers functionality issues arising from manufacturing defects.

How to Claim:
1. Contact support with proof of purchase
2. Provide description of the defect
3. Ship the item for evaluation (if required)
4. Receive replacement or refund

Exclusions:
- Damage from accidents or misuse
- Normal wear and tear
- Software issues
- Water damage (unless waterproof rated)""",
                "last_updated": "2025-05-20",
                "version": "2.0",
                "applicable_regions": ["USA", "Canada"]
            },
            {
                "id": "PAY-001",
                "title": "Accepted Payment Methods",
                "category": "Payment",
                "content": """We accept the following payment methods for your convenience:

Credit and Debit Cards:
- Visa
- Mastercard
- American Express
- Discover

Digital Wallets:
- Apple Pay
- Google Pay
- PayPal

Bank Transfers:
- ACH transfers for orders over $500

Buy Now, Pay Later:
- Afterpay (4 interest-free payments)
- Klarna (flexible payment plans)

Security:
All payment information is encrypted using 256-bit SSL technology. We never store your complete credit card information on our servers.

Payment Processing:
Your payment is authorized immediately but not charged until your order ships.""",
                "last_updated": "2025-08-01",
                "version": "2.2",
                "applicable_regions": ["USA", "Canada"]
            },
            {
                "id": "CANC-001",
                "title": "Order Cancellation Policy",
                "category": "Cancellation",
                "content": """You can cancel your order under the following conditions:

Cancellation Window:
- Orders can be cancelled within 24 hours of purchase
- If order has already shipped, it cannot be cancelled but can be refused

How to Cancel:
1. Log into your account
2. Go to Orders section
3. Click Cancel next to the order
4. Or contact support for assistance

Refunds:
- Cancellations within 24 hours: Full refund including shipping
- After processing begins: Refund minus $5 processing fee

Note:
Pre-ordered or custom items may have different cancellation policies.""",
                "last_updated": "2025-07-10",
                "version": "1.5",
                "applicable_regions": ["USA", "Canada"]
            },
            {
                "id": "EXC-001",
                "title": "Exchange Policy",
                "category": "Exchange",
                "content": """Exchange items within 60 days for size, color, or style differences.

Eligibility:
- Item must be in original condition
- Must be returned within 60 days of purchase
- Only one exchange per item

Process:
1. Initiate exchange request through your account
2. Receive prepaid return shipping label
3. Ship the item to our warehouse
4. Upon receipt of return, new item ships immediately
5. No additional shipping costs for exchanges

Size/Fit Exchanges:
For clothing items, we'll ship the new size before receiving the return (expedited process).""",
                "last_updated": "2025-06-01",
                "version": "1.0",
                "applicable_regions": ["USA"]
            },
            {
                "id": "PRIV-001",
                "title": "Privacy and Data Protection Policy",
                "category": "Privacy",
                "content": """We are committed to protecting your personal information.

Data We Collect:
- Account information (name, email, address, phone)
- Order history and preferences
- Payment information (processed by secure third-party)
- Device and browser information

How We Use Your Data:
- To process and ship your orders
- To send order updates and marketing emails
- To improve our website and services
- To detect and prevent fraud

Your Rights:
- You can update your information anytime in your account
- You can opt out of marketing emails
- You can request a copy of your data
- You can request deletion of your data (exceptions apply)

Data Security:
All data is encrypted in transit and at rest. We use industry-standard security measures.""",
                "last_updated": "2025-08-15",
                "version": "2.1",
                "applicable_regions": ["USA", "Canada", "International"]
            },
            {
                "id": "GEO-001",
                "title": "Geographic Service Areas",
                "category": "Service",
                "content": """We currently ship to the following regions:

Domestic:
- All 50 USA states
- Washington DC

International:
- Canada (all provinces)
- Mexico
- Select European countries

Restrictions:
- Some items may not ship to certain regions due to regulations
- Hazardous materials have limited shipping options
- Weapons and certain equipment cannot be shipped internationally

For shipping costs and timeframes by region, see our Shipping Policy.""",
                "last_updated": "2025-07-20",
                "version": "1.3",
                "applicable_regions": ["USA", "Canada", "Mexico"]
            },
            {
                "id": "QUAL-001",
                "title": "Quality Assurance Standards",
                "category": "Quality",
                "content": """We maintain high quality standards for all products.

Quality Checks:
- All items inspected before packing
- Electronics tested for functionality
- Packaging checked for damage
- Random inspections throughout the month

Product Standards:
- All items meet or exceed industry standards
- Defective items are identified and removed
- Shelf life items checked for expiration
- Counterfeit items never accepted

Defective Items:
If you receive a defective item, contact us immediately. We'll provide a replacement at no cost.""",
                "last_updated": "2025-05-15",
                "version": "1.0",
                "applicable_regions": ["USA", "Canada"]
            },
            {
                "id": "GFT-001",
                "title": "Gift Card Policy",
                "category": "Promotions",
                "content": """Gift cards are a great way to give the gift of choice.

Purchase:
- Denominations: $25, $50, $100, $250
- Email or physical delivery available
- No purchase fees

Usage:
- Valid for 5 years from purchase date
- Can be combined with one promotional code
- Non-refundable after purchase
- Account credit may not be refunded

Lost or Stolen:
- We can assist with lost gift cards if you have the receipt
- Replacement available for a $5 fee
- Without receipt, balance cannot be recovered""",
                "last_updated": "2025-06-01",
                "version": "1.2",
                "applicable_regions": ["USA", "Canada"]
            },
            {
                "id": "BULK-001",
                "title": "Bulk Order Discount Policy",
                "category": "Promotions",
                "content": """We offer special pricing for bulk orders.

Quantity Discounts:
- 10-25 items: 10% discount
- 26-50 items: 15% discount
- 51-100 items: 20% discount
- 100+ items: Custom pricing (contact sales)

Process:
1. Contact our sales team with your order details
2. Receive a custom quote within 24 hours
3. Process your bulk order with expedited support

Eligibility:
- Minimum order value: $500
- Orders typically ship within 3-5 business days
- Volume pricing excludes current promotions""",
                "last_updated": "2025-07-15",
                "version": "1.0",
                "applicable_regions": ["USA", "Canada"]
            },
            {
                "id": "ACC-001",
                "title": "Accessibility and Accommodations",
                "category": "Service",
                "content": """We're committed to making our services accessible to everyone.

Website Accessibility:
- WCAG 2.1 Level AA compliant
- Screen reader compatible
- Keyboard navigation support
- High contrast mode available

Customer Support Accommodations:
- TTY/TDD support for hearing impaired
- Large print options
- Alternative format materials
- Extended response times if needed

Assistance:
If you need accommodations, please contact our accessibility team at accessibility@company.com with your specific needs.""",
                "last_updated": "2025-08-10",
                "version": "1.1",
                "applicable_regions": ["USA", "Canada"]
            },
            {
                "id": "ENV-001",
                "title": "Environmental and Sustainability Policy",
                "category": "Corporate",
                "content": """We are committed to environmental responsibility.

Packaging:
- 100% recyclable packaging materials
- Minimized packaging waste
- Eco-friendly shipping materials
- Reduced plastic usage

Operations:
- Carbon-neutral shipping available
- Renewable energy in warehouses
- Waste reduction programs
- Sustainable product sourcing

Customer Participation:
- We encourage customers to recycle
- Take-back programs for electronics
- Donations to environmental organizations
- Partner with carbon offset providers""",
                "last_updated": "2025-06-20",
                "version": "1.0",
                "applicable_regions": ["USA", "Canada"]
            },
            {
                "id": "ACC-002",
                "title": "Accessibility Statement",
                "category": "Service",
                "content": """This accessibility statement applies to our website and mobile application.

Commitment:
We are committed to providing a website and mobile experience that is accessible to everyone, including individuals with disabilities.

Accessibility Features:
- Text alternatives for images
- Resizable text
- Clear navigation
- Form field labels
- Keyboard accessible

Feedback:
If you encounter any accessibility issues or need assistance, please contact us immediately at support@company.com with detailed information.""",
                "last_updated": "2025-08-12",
                "version": "1.0",
                "applicable_regions": ["USA", "Canada"]
            },
            {
                "id": "DMG-001",
                "title": "Damage Claims and Resolution",
                "category": "Returns",
                "content": """If your item arrives damaged, we'll make it right.

Damage Claims Process:
1. Unbox item carefully and inspect immediately
2. Document damage with photos
3. Contact support within 48 hours of delivery
4. Provide photos and order number

Resolution Options:
- Full refund (return shipping prepaid)
- Replacement shipment
- Store credit

Documentation Required:
- Photos of damaged item and packaging
- Order confirmation
- Tracking number

Timeline:
Claims must be filed within 7 days of delivery. We'll resolve most claims within 5-7 business days.""",
                "last_updated": "2025-07-25",
                "version": "1.2",
                "applicable_regions": ["USA", "Canada"]
            },
            {
                "id": "LOY-001",
                "title": "Loyalty Program Terms",
                "category": "Promotions",
                "content": """Our loyalty program rewards your continued business.

Tier Benefits:
- Bronze: 1% cash back, free shipping
- Silver: 3% cash back, priority support
- Gold: 5% cash back, exclusive sales access
- Platinum: 7% cash back, VIP support, free expedited shipping

Earning:
- 1 point per $1 spent
- Bonus points on select items
- Points earned immediately after purchase
- No expiration on points

Redemption:
- Redeem points for discounts or free items
- Minimum redemption: 500 points ($5)
- Can combine with select promotions""",
                "last_updated": "2025-08-01",
                "version": "2.0",
                "applicable_regions": ["USA", "Canada"]
            },
            {
                "id": "INTL-001",
                "title": "International Shipping and Customs",
                "category": "Shipping",
                "content": """Information about international orders and customs.

Shipping Destinations:
- Canada: 5-10 business days
- Mexico: 7-14 business days
- European Union: 10-21 business days
- Other countries: Custom timeframes

Customs and Duties:
- Customers responsible for any customs fees
- All items declare declared value
- Some items may be restricted
- Delivery delays possible due to customs

International Returns:
- Return shipping costs vary by country
- Contact support for international return label
- Process same as domestic returns
- May take longer due to customs

Restrictions:
- Hazardous materials not eligible
- Weapons and certain items restricted
- Liquids and batteries have limitations""",
                "last_updated": "2025-07-30",
                "version": "1.1",
                "applicable_regions": ["USA", "Canada", "International"]
            },
            {
                "id": "PRICE-001",
                "title": "Price Match Guarantee",
                "category": "Promotions",
                "content": """We want you to get the best price.

Price Match Policy:
- We'll match competitor prices on identical items
- Match applies to major retailers
- Price must be current and in-stock
- Match applies to item price only (excludes shipping)

How to Request:
1. Find the lower price online
2. Contact us with the product and competitor link
3. Provide proof of the price
4. We'll apply the price match or offer alternatives

Exclusions:
- Clearance/Final sale items
- Items from unauthorized sellers
- Pre-order or special order items
- Prices that appear to be errors

Timeframe:
- Requests processed within 48 hours
- Match applied to new orders or as store credit""",
                "last_updated": "2025-08-05",
                "version": "1.0",
                "applicable_regions": ["USA", "Canada"]
            },
            {
                "id": "CON-001",
                "title": "Consumer Protection Compliance",
                "category": "Legal",
                "content": """We comply with all applicable consumer protection laws.

Consumer Rights:
- Right to know what we collect about you
- Right to dispute charges
- Right to cancel orders within policy
- Right to safe and legal products

Fair Trading:
- Honest product descriptions
- Clear pricing and terms
- No hidden fees
- Transparent return policy

Dispute Resolution:
- Contact support first
- File claim if unresolved
- Escalate to management
- External dispute resolution available

Compliance:
- FTC Safeguards Rule compliance
- GDPR compliance for EU customers
- CCPA compliance for California residents
- State-specific consumer protection laws""",
                "last_updated": "2025-06-30",
                "version": "1.0",
                "applicable_regions": ["USA", "Canada", "International"]
            },
            {
                "id": "SEC-001",
                "title": "Security and Fraud Prevention",
                "category": "Security",
                "content": """We take security seriously to protect your information.

Security Measures:
- 256-bit SSL encryption
- PCI DSS compliance
- Regular security audits
- Fraud detection systems

Your Responsibilities:
- Keep password confidential
- Do not share account information
- Monitor account for unauthorized activity
- Report suspicious behavior

Fraud Protection:
- Zero liability for unauthorized purchases
- Dispute resolution process available
- Chargeback support provided
- Investigation of fraud reports

Reporting Fraud:
- Contact us immediately if suspicious
- Do not make additional purchases
- Provide details of fraudulent activity
- Law enforcement may be involved""",
                "last_updated": "2025-08-10",
                "version": "1.0",
                "applicable_regions": ["USA", "Canada", "International"]
            }
        ]

        self.policies = policies_data

    def _load_faqs(self):
        """Load FAQ database (30+ frequently asked questions)."""
        faqs_data = [
            {
                "id": "FAQ-001",
                "category": "Orders",
                "question": "How can I track my order?",
                "answer": "Once your order ships, you'll receive an email with a tracking number. You can use this number on the carrier's website to track your package in real-time.",
                "helpful_count": 1240,
                "view_count": 5890,
                "last_updated": "2025-08-10"
            },
            {
                "id": "FAQ-002",
                "category": "Returns",
                "question": "How long do I have to return an item?",
                "answer": "You have 30 days from the date of purchase to return most items. Some items may have different return windows - check the product page for details.",
                "helpful_count": 2150,
                "view_count": 8920,
                "last_updated": "2025-08-12"
            },
            {
                "id": "FAQ-003",
                "category": "Shipping",
                "question": "Do you offer free shipping?",
                "answer": "Yes! Orders over $50 qualify for free standard shipping (5-7 business days). Express and overnight options are available for an additional fee.",
                "helpful_count": 1890,
                "view_count": 7650,
                "last_updated": "2025-08-08"
            },
            {
                "id": "FAQ-004",
                "category": "Payment",
                "question": "What payment methods do you accept?",
                "answer": "We accept Visa, Mastercard, American Express, Discover, Apple Pay, Google Pay, PayPal, and select buy-now-pay-later options.",
                "helpful_count": 1520,
                "view_count": 6200,
                "last_updated": "2025-08-09"
            },
            {
                "id": "FAQ-005",
                "category": "Account",
                "question": "How do I reset my password?",
                "answer": "Click 'Forgot Password' on the login page and enter your email. You'll receive a reset link via email within 5 minutes.",
                "helpful_count": 980,
                "view_count": 4560,
                "last_updated": "2025-08-07"
            },
            {
                "id": "FAQ-006",
                "category": "Products",
                "question": "Are your products genuine?",
                "answer": "Yes, all our products are 100% authentic. We source directly from manufacturers and authorized distributors to ensure quality.",
                "helpful_count": 845,
                "view_count": 3890,
                "last_updated": "2025-08-06"
            },
            {
                "id": "FAQ-007",
                "category": "Shipping",
                "question": "How much does shipping cost?",
                "answer": "Shipping costs depend on your location and method chosen. Standard shipping is free on orders over $50, or $5.99 for smaller orders.",
                "helpful_count": 2340,
                "view_count": 9120,
                "last_updated": "2025-08-11"
            },
            {
                "id": "FAQ-008",
                "category": "Returns",
                "question": "Will I have to pay for return shipping?",
                "answer": "No! We provide a prepaid return shipping label for all eligible returns. Simply print the label and drop off at any carrier location.",
                "helpful_count": 1670,
                "view_count": 5890,
                "last_updated": "2025-08-10"
            },
            {
                "id": "FAQ-009",
                "category": "Orders",
                "question": "Can I change my order after placing it?",
                "answer": "If your order hasn't shipped yet, you can contact us to modify it within 24 hours of purchase. After that, you'll need to place a new order.",
                "helpful_count": 920,
                "view_count": 4120,
                "last_updated": "2025-08-09"
            },
            {
                "id": "FAQ-010",
                "category": "Account",
                "question": "How can I update my profile information?",
                "answer": "Log into your account, go to Account Settings, and update your personal, billing, and shipping information as needed.",
                "helpful_count": 710,
                "view_count": 3450,
                "last_updated": "2025-08-08"
            },
            {
                "id": "FAQ-011",
                "category": "Warranty",
                "question": "What warranty coverage comes with products?",
                "answer": "Most products come with 6-12 month manufacturer warranty covering defects. Electronics get 12 months, accessories get 6 months.",
                "helpful_count": 1340,
                "view_count": 5620,
                "last_updated": "2025-08-12"
            },
            {
                "id": "FAQ-012",
                "category": "Promotions",
                "question": "How do I use a coupon code?",
                "answer": "Enter your coupon code in the 'Promo Code' field at checkout before completing your purchase. The discount will apply immediately.",
                "helpful_count": 1560,
                "view_count": 6890,
                "last_updated": "2025-08-11"
            },
            {
                "id": "FAQ-013",
                "category": "Products",
                "question": "Do you have international products?",
                "answer": "Yes, we ship internationally to Canada, Mexico, and select European countries. Shipping costs and times vary by location.",
                "helpful_count": 890,
                "view_count": 4230,
                "last_updated": "2025-08-10"
            },
            {
                "id": "FAQ-014",
                "category": "Orders",
                "question": "When will my order arrive?",
                "answer": "Delivery times vary by shipping method: Standard (5-7 days), Express (2-3 days), Overnight (1 day). Orders ship within 1-2 business days of purchase.",
                "helpful_count": 2890,
                "view_count": 11200,
                "last_updated": "2025-08-13"
            },
            {
                "id": "FAQ-015",
                "category": "Account",
                "question": "How do I delete my account?",
                "answer": "Contact our support team to request account deletion. We'll remove your personal information within 30 days as required by law.",
                "helpful_count": 450,
                "view_count": 2340,
                "last_updated": "2025-08-09"
            },
            {
                "id": "FAQ-016",
                "category": "Shipping",
                "question": "Do you ship to PO boxes?",
                "answer": "No, we only ship to physical addresses. However, many PO box services provide street addresses that we can use.",
                "helpful_count": 620,
                "view_count": 3120,
                "last_updated": "2025-08-08"
            },
            {
                "id": "FAQ-017",
                "category": "Returns",
                "question": "What items are non-returnable?",
                "answer": "Clearance/Final Sale items, customized/personalized products, and items purchased more than 30 days ago cannot be returned.",
                "helpful_count": 1240,
                "view_count": 5890,
                "last_updated": "2025-08-12"
            },
            {
                "id": "FAQ-018",
                "category": "Payment",
                "question": "Is my payment information secure?",
                "answer": "Yes, we use 256-bit SSL encryption to protect all payment data. We never store your complete credit card information on our servers.",
                "helpful_count": 1890,
                "view_count": 7520,
                "last_updated": "2025-08-11"
            },
            {
                "id": "FAQ-019",
                "category": "Products",
                "question": "Are there any restrictions on product shipping?",
                "answer": "Some items (hazardous materials, weapons) have shipping restrictions. The product page will indicate if an item can't ship to your location.",
                "helpful_count": 780,
                "view_count": 3890,
                "last_updated": "2025-08-10"
            },
            {
                "id": "FAQ-020",
                "category": "Orders",
                "question": "Can I combine multiple orders?",
                "answer": "Unfortunately, orders are processed separately. However, if items are in stock, combining items in one order usually results in faster delivery.",
                "helpful_count": 540,
                "view_count": 2890,
                "last_updated": "2025-08-09"
            },
            {
                "id": "FAQ-021",
                "category": "Account",
                "question": "Do you offer a loyalty program?",
                "answer": "Yes! Our tiered loyalty program (Bronze, Silver, Gold, Platinum) offers exclusive benefits and discounts based on your spending.",
                "helpful_count": 1450,
                "view_count": 6780,
                "last_updated": "2025-08-13"
            },
            {
                "id": "FAQ-022",
                "category": "Shipping",
                "question": "What is the difference between Express and Overnight shipping?",
                "answer": "Express shipping (2-3 business days) is $12.99, while Overnight shipping (1 business day) is $24.99. Both include tracking.",
                "helpful_count": 1120,
                "view_count": 4560,
                "last_updated": "2025-08-12"
            },
            {
                "id": "FAQ-023",
                "category": "Returns",
                "question": "How long does a refund take?",
                "answer": "After we receive and inspect your return, your refund will be processed within 5-7 business days. Return shipping takes 3-5 days.",
                "helpful_count": 1680,
                "view_count": 6200,
                "last_updated": "2025-08-11"
            },
            {
                "id": "FAQ-024",
                "category": "Products",
                "question": "Do you price match competitors?",
                "answer": "We strive to offer competitive pricing. If you find a lower price, contact us with details for a potential price adjustment.",
                "helpful_count": 890,
                "view_count": 4120,
                "last_updated": "2025-08-10"
            },
            {
                "id": "FAQ-025",
                "category": "Payment",
                "question": "Do you offer payment plans?",
                "answer": "Yes, we offer Afterpay and Klarna for flexible payment options. Select these at checkout for payment plans.",
                "helpful_count": 1240,
                "view_count": 5890,
                "last_updated": "2025-08-12"
            },
            {
                "id": "FAQ-026",
                "category": "Orders",
                "question": "What if I don't receive my order?",
                "answer": "Contact us immediately if your order doesn't arrive by the estimated delivery date. We'll investigate with the carrier and send a replacement.",
                "helpful_count": 1560,
                "view_count": 6450,
                "last_updated": "2025-08-13"
            },
            {
                "id": "FAQ-027",
                "category": "Account",
                "question": "How can I contact customer support?",
                "answer": "You can reach us via email (support@company.com), live chat (9am-5pm EST), or phone (1-800-SUPPORT) Monday-Friday.",
                "helpful_count": 2340,
                "view_count": 9120,
                "last_updated": "2025-08-14"
            },
            {
                "id": "FAQ-028",
                "category": "Returns",
                "question": "Can I exchange an item?",
                "answer": "Yes, you can exchange items within 60 days for size, color, or style differences. See our Exchange Policy for details.",
                "helpful_count": 1120,
                "view_count": 5340,
                "last_updated": "2025-08-12"
            },
            {
                "id": "FAQ-029",
                "category": "Promotions",
                "question": "When do you have sales?",
                "answer": "We run promotions seasonally and offer flash sales weekly. Subscribe to our newsletter for the latest deals and exclusive discounts.",
                "helpful_count": 1890,
                "view_count": 7890,
                "last_updated": "2025-08-13"
            },
            {
                "id": "FAQ-030",
                "category": "Products",
                "question": "How are product reviews verified?",
                "answer": "Reviews are from verified purchasers only. We remove any inappropriate or fraudulent reviews to maintain integrity.",
                "helpful_count": 720,
                "view_count": 3560,
                "last_updated": "2025-08-11"
            }
        ]

        self.faqs = faqs_data

    def _load_support_tickets(self):
        """Load support ticket data."""
        subjects = [
            "Order not received",
            "Defective product",
            "Wrong item shipped",
            "Return authorization",
            "Refund status",
            "Shipping delay",
            "Login issue",
            "Password reset",
            "Account security",
            "Billing inquiry"
        ]

        resolutions = [
            "Replacement shipped",
            "Full refund processed",
            "Escalated to management",
            "Issue resolved",
            "Awaiting customer response",
            "Item reshipped",
            "Credit issued",
            "Password reset link sent",
            "Account secured",
            "Refund pending"
        ]

        base_date = datetime.now() - timedelta(days=90)
        statuses = ["open", "in_progress", "resolved", "closed"]
        priorities = ["low", "medium", "high", "urgent"]

        for i in range(50):
            ticket_id = f"TKT-{50000 + i}"
            customer_id = random.choice(list(self.customers.keys()))
            created_at = base_date + timedelta(days=random.randint(0, 90))
            updated_at = created_at + timedelta(days=random.randint(0, 10))

            self.support_tickets[ticket_id] = {
                "id": ticket_id,
                "customer_id": customer_id,
                "created_at": created_at.isoformat(),
                "updated_at": updated_at.isoformat(),
                "status": random.choice(statuses),
                "priority": random.choice(priorities),
                "subject": random.choice(subjects),
                "description": "Customer reported an issue with their recent order. Investigation in progress.",
                "resolution": random.choice(resolutions) if random.random() > 0.3 else None,
                "category": random.choice(["Orders", "Returns", "Shipping", "Account", "Billing"])
            }


# ============================================================================
# Database API Functions
# ============================================================================

# Global database instance
_db = CustomerChatbotDatabase()


def _normalize_record_id(record_id: Optional[str]) -> str:
    """Normalize a user-supplied ID to canonical form (e.g. 'cust 10000' -> 'CUST-10000')."""
    import re

    if not record_id:
        return ""
    cleaned = str(record_id).strip()
    match = re.match(r"^\s*([a-zA-Z]+)\s*-?\s*(\d+)\s*$", cleaned)
    if match:
        return f"{match.group(1).upper()}-{match.group(2)}"
    return cleaned.upper()


# ---------------------------------------------------------------------------
# Where the records come from
# ---------------------------------------------------------------------------
#
# By default they are generated above, in memory, from a fixed seed. If a
# `customer.db` exists they are read from SQLite instead -- same functions,
# same dicts out, so nothing above this line notices which it was.
#
# The fallback is not a nicety: a fresh clone and CI have no database file,
# and every one of the ~300 tests must keep passing without one. `active()`
# returning None is the normal case, never an error.


def _store():
    """The SQLite store if one is configured and readable, else None."""
    import customer_db

    return customer_db.active()


def get_customer_by_id(customer_id: str) -> Optional[Dict[str, Any]]:
    """Get customer by ID (case-insensitive)."""
    normalized = _normalize_record_id(customer_id)
    store = _store()
    if store is not None:
        return store.customer_by_id(normalized)
    return _db.customers.get(normalized)


def get_customer_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Find a customer by email address (case-insensitive).

    Gives customers who don't have their ID to hand a second way in.
    """
    if not email:
        return None
    target = str(email).strip().lower()
    store = _store()
    if store is not None:
        return store.customer_by_email(target)
    for customer in _db.customers.values():
        if (customer.get("email") or "").lower() == target:
            return customer
    return None


def get_orders_by_customer(customer_id: str) -> List[Dict[str, Any]]:
    """Get all orders for a customer (case-insensitive)."""
    normalized = _normalize_record_id(customer_id)
    store = _store()
    if store is not None:
        return store.orders_for(normalized)
    return [order for order in _db.orders.values() if order["customer_id"] == normalized]


def get_order_details(order_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about an order (case-insensitive)."""
    normalized = _normalize_record_id(order_id)
    store = _store()
    if store is not None:
        return store.order(normalized)
    return _db.orders.get(normalized)


def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics."""
    pending_orders = sum(1 for order in _db.orders.values() if order["status"] == "processing")
    pending_tickets = sum(1 for ticket in _db.support_tickets.values() if ticket["status"] in ["open", "in_progress"])

    total_tickets = len(_db.support_tickets)
    resolved_tickets = sum(1 for ticket in _db.support_tickets.values() if ticket["status"] == "resolved")
    satisfaction_score = 4.6 if total_tickets == 0 else round(4.2 + (resolved_tickets / total_tickets) * 0.5, 1)

    return {
        "timestamp": datetime.now().isoformat(),
        "system_status": "healthy",
        "database_connection": "connected",
        "active_sessions": random.randint(25, 75),
        "api_response_time_ms": round(random.uniform(80, 180), 1),
        "cache_hit_rate": round(random.uniform(0.80, 0.95), 2),
        "pending_orders": pending_orders,
        "pending_tickets": pending_tickets,
        "customer_satisfaction_score": satisfaction_score,
        "uptime_hours": random.randint(500, 1000),
        "error_rate_percent": round(random.uniform(0.01, 0.05), 2)
    }


# ============================================================================
# Convenience Functions for Testing
# ============================================================================

def get_all_customers() -> List[Dict[str, Any]]:
    """Get all customers."""
    store = _store()
    if store is not None:
        return store.all_customers()
    return list(_db.customers.values())


def get_all_orders() -> List[Dict[str, Any]]:
    """Get all orders."""
    store = _store()
    if store is not None:
        return store.all_orders()
    return list(_db.orders.values())


def get_all_products() -> List[Dict[str, Any]]:
    """Get all products."""
    return list(_db.products.values())


def get_all_faqs(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get FAQs, optionally filtered by category."""
    if category:
        return [faq for faq in _db.faqs if faq["category"].lower() == category.lower()]
    return _db.faqs


def get_all_policies() -> List[Dict[str, Any]]:
    """Get all policies."""
    return _db.policies


def get_all_support_tickets() -> List[Dict[str, Any]]:
    """Get all support tickets."""
    return list(_db.support_tickets.values())
