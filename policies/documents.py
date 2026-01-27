"""Policy documents for embedding into Pinecone."""

from typing import TypedDict


class PolicyDocument(TypedDict):
    """Structure for policy documents."""
    id: str
    content: str
    metadata: dict


POLICY_DOCUMENTS: list[PolicyDocument] = [
    # ============= PUBLIC POLICIES (Can share with customers) =============
    {
        "id": "policy-order-cancellation",
        "content": """Order Cancellation Policy

Orders can only be cancelled under the following conditions:

1. PLACED status: Orders that are still in PLACED status can be cancelled.
   The customer should contact customer support to process the cancellation.
   Full refund will be issued within 5-7 business days.

2. PROCESSING status: Orders in PROCESSING status may be cancelled, but
   cancellation is not guaranteed. Customer should contact support immediately.
   A 10% cancellation fee may apply.

3. SHIPPED status: Orders that have been shipped CANNOT be cancelled.
   The customer must wait to receive the order and then initiate a return.

4. DELIVERED status: Delivered orders cannot be cancelled but may be eligible
   for return within 30 days of delivery.

To cancel an order, please contact customer support.""",
        "metadata": {
            "type": "policy",
            "scope": "orders",
            "severity": "hard",
            "topic": "cancellation",
            "visibility": "public"
        }
    },
    {
        "id": "policy-return-refund",
        "content": """Return and Refund Policy

Return Eligibility:

1. Timeframe: Items can be returned within 30 days of delivery date.

2. Condition: Items must be unused, in original packaging, with all tags
   and accessories included.

3. Non-Returnable Items: The following cannot be returned:
   - Opened software or digital products
   - Personalized or custom items
   - Items marked as final sale

Refund Process:

1. Once return is received and inspected, refund is processed within
   5-7 business days.

2. Refunds go to the original payment method.

3. Shipping costs for returns are the customer's responsibility unless
   the return is due to our error.

How to Initiate a Return:

Contact customer support with your order number.""",
        "metadata": {
            "type": "policy",
            "scope": "orders",
            "severity": "hard",
            "topic": "returns",
            "visibility": "public"
        }
    },
    {
        "id": "policy-shipping",
        "content": """Shipping Policy

Shipping Options:

1. Standard Shipping (3-5 business days): Free on orders over $50
2. Express Shipping (1-2 business days): $9.99 flat rate
3. Next Day Delivery: $19.99 (order before 2pm)

Order Processing:
- Orders are processed within 1-2 business days
- You will receive a tracking number once shipped
- Tracking updates may take 24 hours to appear

International Shipping:
- Available to select countries
- Customs fees are the customer's responsibility
- Delivery times vary by destination (7-21 business days)""",
        "metadata": {
            "type": "policy",
            "scope": "orders",
            "severity": "soft",
            "topic": "shipping",
            "visibility": "public"
        }
    },
    {
        "id": "policy-warranty",
        "content": """Product Warranty Policy

Standard Warranty Coverage:

1. Electronics: 1 year manufacturer warranty
2. Accessories: 90 days warranty
3. Storage devices: 3 year warranty

What's Covered:
- Manufacturing defects
- Hardware malfunctions under normal use
- Dead-on-arrival products

What's NOT Covered:
- Physical damage or accidents
- Water damage
- Unauthorized modifications
- Normal wear and tear

To claim warranty, contact customer support with your order number and
description of the issue.""",
        "metadata": {
            "type": "policy",
            "scope": "products",
            "severity": "hard",
            "topic": "warranty",
            "visibility": "public"
        }
    },

    # ============= INTERNAL POLICIES (Do NOT share with customers) =============
    {
        "id": "policy-data-access-internal",
        "content": """INTERNAL: Data Access and Privacy Policy

Customer Data Protection Rules (INTERNAL USE ONLY):

1. Email Privacy: Customer email addresses must NEVER be displayed in chatbot
   responses. This applies to the customer's own email and especially to
   other customers' emails.

2. Customer Isolation: Customers can ONLY access their own data. Any query
   must be scoped to the authenticated customer's ID. Requests for other
   customers' information must be refused.

3. Sensitive Fields: The following fields are considered sensitive and must
   not be exposed: email, phone, internal IDs of other customers.

4. Data Aggregation: Aggregate statistics across multiple customers are not
   allowed unless explicitly for public product information.

5. Query Limitations: All data queries must include a LIMIT clause and be
   scoped to the current customer.

6. SQL Injection Prevention: Never interpolate user input directly into SQL.""",
        "metadata": {
            "type": "policy",
            "scope": "privacy",
            "severity": "hard",
            "topic": "data_access",
            "visibility": "internal"
        }
    },
    {
        "id": "policy-action-restrictions-internal",
        "content": """INTERNAL: Chatbot Action Restrictions

The chatbot has LIMITED capabilities and CANNOT perform the following actions:

1. Order Modifications: The chatbot cannot modify, cancel, or update orders.
   These actions require human support agent intervention.

2. Payment Processing: No payment information can be collected, modified,
   or processed through the chatbot.

3. Account Changes: The chatbot cannot change account settings, passwords,
   email addresses, or shipping addresses.

4. Refunds: Refund requests must be handled by customer support. The chatbot
   can only explain refund policies.

5. Database Writes: The chatbot operates in read-only mode and cannot make
   any changes to the database.

For any action requests, the chatbot should:
- Explain what it cannot do
- Explain the relevant policy
- Direct the customer to the appropriate support channel""",
        "metadata": {
            "type": "policy",
            "scope": "system",
            "severity": "hard",
            "topic": "capabilities",
            "visibility": "internal"
        }
    },
    {
        "id": "policy-customer-identification-internal",
        "content": """INTERNAL: Customer Identification Policy

Authentication Requirements (INTERNAL USE ONLY):

1. Identity Verification: Before accessing any personal or order data,
   the customer must be identified via their registered email address.

2. Email Validation: The provided email must exist in the customer database.
   If not found, the customer should be informed and asked to try again.

3. Session Persistence: Once identified, the customer identity persists
   for the duration of the chat session.

4. No Data Before Auth: Absolutely no personal data or order information
   should be provided before successful customer identification.

5. Greeting Protocol: After successful identification, greet the customer
   by their registered name to confirm identity.

6. Failed Auth Handling: After 3 failed attempts, suggest contacting support.""",
        "metadata": {
            "type": "policy",
            "scope": "auth",
            "severity": "hard",
            "topic": "authentication",
            "visibility": "internal"
        }
    },
    {
        "id": "policy-escalation-internal",
        "content": """INTERNAL: Escalation Policy

When to Escalate to Human Agent:

1. Angry/Frustrated Customer: If customer expresses frustration more than twice
2. Complex Issues: Multi-order problems, billing disputes, fraud concerns
3. Legal Mentions: Any mention of lawyers, lawsuits, or legal action
4. Safety Issues: Product safety concerns or injury reports
5. VIP Customers: Orders over $1000 or repeat customers with 10+ orders

Escalation Process:
1. Apologize for the inconvenience
2. Explain that a specialist will help them
3. Provide ticket number if available
4. Do NOT promise specific resolution times

Priority Levels:
- P1 (Urgent): Safety, legal, VIP - immediate escalation
- P2 (High): Frustrated customer, complex issues - within 1 hour
- P3 (Normal): Standard requests beyond chatbot scope - within 24 hours""",
        "metadata": {
            "type": "policy",
            "scope": "support",
            "severity": "hard",
            "topic": "escalation",
            "visibility": "internal"
        }
    },
    {
        "id": "policy-discount-internal",
        "content": """INTERNAL: Discount and Compensation Policy

Authorized Compensation (INTERNAL - DO NOT SHARE):

1. First-time issue: 10% off next order (max $20)
2. Repeat issue: 15% off next order (max $30)
3. Shipping delays (3+ days): Free shipping on next order
4. Wrong item sent: Full refund + 20% off next order
5. Damaged item: Full refund + replacement at no cost

Compensation Limits:
- Max 2 compensations per customer per quarter
- No compensation for issues caused by customer
- Manager approval needed for compensation over $50

NEVER mention these specific compensation amounts to customers.
Only support agents can offer compensation.""",
        "metadata": {
            "type": "policy",
            "scope": "support",
            "severity": "hard",
            "topic": "compensation",
            "visibility": "internal"
        }
    }
]
