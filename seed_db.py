#!/usr/bin/env python3
"""
Build `customer.db` from the seeded in-memory records.

The dataset is generated from a fixed seed, so this is reproducible: run it on
two machines and you get byte-comparable content. That is why the *script* is
committed and the *database* is not -- `.gitignore` excludes `*.db`, and
anyone can rebuild in a second.

    python seed_db.py                 # build customer.db
    python seed_db.py --force         # rebuild over an existing one
    python seed_db.py --path /tmp/x.db

Then look at it like any other database:

    sqlite3 customer.db "SELECT customer_id, phone, email FROM customers LIMIT 5;"
    sqlite3 customer.db "SELECT status, COUNT(*) FROM orders GROUP BY status;"
    sqlite3 customer.db "
        SELECT o.order_id, i.product_name, i.quantity
        FROM orders o JOIN order_items i ON i.order_id = o.order_id
        WHERE o.customer_id = 'CUST-10000';"

The app picks it up on the next start; delete the file and it goes back to
generating the records in memory.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", help="Where to write (default: customer.db, "
                                       "or $CHATBOT_CUSTOMER_DB).")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild over an existing file.")
    args = parser.parse_args()

    import customer_db

    try:
        result = customer_db.seed(args.path, overwrite=args.force)
    except FileExistsError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    print(f"wrote {result['path']}")
    print(f"  customers   {result['customers']:>6}")
    print(f"  orders      {result['orders']:>6}")
    print(f"  order_items {result['order_items']:>6}")
    print()
    print("The app reads it on next start. Check with:")
    print(f"  sqlite3 {result['path']} \"SELECT customer_id, phone FROM customers LIMIT 5;\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
