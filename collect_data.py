import psycopg2
import json
import random
import statistics
import time


def get_connection():
    return psycopg2.connect(dbname="dvdrental", user="postgres", host="127.0.0.1")

def get_random_deep_sql():
    # 1. Full Schema Map based on real Database Relationships
    # This prevents "Cross-Joins" that would freeze your collection.
    full_schema = {
        "actor": ["film_actor"],
        "film_actor": ["actor", "film"],
        "film": ["film_actor", "film_category", "inventory"],
        "film_category": ["film", "category"],
        "category": ["film_category"],
        "inventory": ["film", "store", "rental"],
        "store": ["inventory", "staff", "address"],
        "staff": ["store", "address", "rental", "payment"],
        "address": ["store", "staff", "customer", "city"],
        "city": ["address", "country"],
        "country": ["city"],
        "customer": ["address", "rental", "payment"],
        "rental": ["inventory", "customer", "staff", "payment"],
        "payment": ["customer", "staff", "rental"]
    }

    # Map of which key to use when joining two specific tables
    join_keys = {
        "actor": "actor_id", "film_actor": "actor_id",
        "film": "film_id", "film_category": "film_id", "inventory": "film_id",
        "category": "category_id",
        "rental": "inventory_id", # inventory join
        "store": "store_id",
        "staff": "staff_id",
        "address": "address_id",
        "city": "city_id",
        "country": "country_id",
        "customer": "customer_id",
        "payment": "rental_id"
    }

    # 2. Complexity Control
    # Depth 1-4 covers simple lookups to deep analytical joins
    depth = random.randint(1, 4)
    start_table = random.choice(list(full_schema.keys()))
    used_tables = [start_table]

    # Use column list or *; USING helps handle duplicate join columns
    query = f"SELECT * FROM {start_table}"

    for _ in range(depth):
        current = used_tables[-1]
        # Find neighbors not already in the query
        options = [t for t in full_schema[current] if t not in used_tables]
        if not options: break

        next_t = random.choice(options)

        # Determine the correct join key
        # If joining film_actor and film, we use film_id
        if next_t in ["film", "film_actor", "film_category", "inventory"]:
            key = "film_id"
        elif next_t in ["actor", "film_actor"]:
            key = "actor_id"
        elif next_t in ["city", "address"]:
            key = "city_id"
        elif next_t in ["country", "city"]:
            key = "country_id"
        else:
            # Fallback to standard naming: table_id
            key = join_keys.get(next_t, f"{next_t}_id")

        query += f" JOIN {next_t} USING ({key})"
        used_tables.append(next_t)

    # 3. Add a "Noise" Filter
    # This randomizes the number of rows Postgres has to process,
    # giving your GNN different 'costs' to learn from.
    if random.random() < 0.5:
        query += f" WHERE random() < {random.uniform(0.1, 0.9)}"

    query += f" LIMIT {random.randint(10, 1000)}"
    return query


def collect(size):
    try:
        conn = get_connection()
        cur = conn.cursor()
        dataset = []
        num_runs = 3

        # Track the extremes
        min_median = float('inf')
        max_median = float('-inf')

        print(f"🚀 Starting collection of {size} queries...")

        for i in range(1, size + 1):

            sql = get_random_deep_sql()
            query_runtimes = []
            query_plans = []

            try:
                for _ in range(num_runs):
                    cur.execute("BEGIN;")
                    cur.execute(f"EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS) {sql}")
                    res = cur.fetchone()[0][0]['Plan']
                    query_runtimes.append(res['Actual Total Time'])
                    query_plans.append(res)
                    cur.execute("ROLLBACK;")

                # Calculate median
                m_time = statistics.median(query_runtimes)

                # Update records
                if m_time < min_median: min_median = m_time
                if m_time > max_median: max_median = m_time

                median_idx = query_runtimes.index(m_time)
                dataset.append({
                    "plan": query_plans[median_idx],
                    "runtime": m_time
                })

            except Exception:
                conn.rollback()
                continue

            # Print status every 100 queries
            if i % 100 == 0:
                print("-" * 50)
                print(f"📊 Progress: {i}/{size}")
                print(f"📉 Lowest Median so far:  {min_median:.4f} ms")
                print(f"📈 Highest Median so far: {max_median:.4f} ms")
                print(f"⏱️  Current Median:        {m_time:.4f} ms")

        with open("query_data.json", "w") as f:
            json.dump(dataset, f)

        print("\n✅ COLLECTION COMPLETE")
        print(f"Final Range: {min_median:.4f} ms to {max_median:.4f} ms")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    collect(20000)