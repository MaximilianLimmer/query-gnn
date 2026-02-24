import psycopg2
import json
import random
import statistics
import time


def get_connection():
    return psycopg2.connect(
        dbname="dvdrental",
        user="postgres_user",   # Changed from 'postgres'
        password="password",    # Added the password
        host="127.0.0.1"
    )
def get_random_deep_sql():
    # COMPLETE DVD-Rental Schema Map
    schema = {
        "film": {"inventory": "film_id", "film_category": "film_id", "film_actor": "film_id"},
        "film_actor": {"film": "film_id", "actor": "actor_id"},
        "actor": {"film_actor": "actor_id"},
        "film_category": {"film": "film_id", "category": "category_id"},
        "category": {"film_category": "category_id"},
        "inventory": {"film": "film_id", "rental": "inventory_id", "store": "store_id"},
        "rental": {"inventory": "inventory_id", "customer": "customer_id", "payment": "rental_id", "staff": "staff_id"},
        "customer": {"rental": "customer_id", "payment": "customer_id", "address": "address_id"},
        "payment": {"rental": "rental_id", "customer": "customer_id", "staff": "staff_id"},
        "address": {"customer": "address_id", "city": "city_id", "staff": "address_id", "store": "address_id"},
        "city": {"address": "city_id", "country": "country_id"},
        "country": {"city": "country_id"},
        "store": {"staff": "manager_staff_id", "inventory": "store_id", "address": "address_id"},
        "staff": {"store": "store_id", "payment": "staff_id", "rental": "staff_id", "address": "address_id"}
    }

    if random.random() < 0.15:
        # --- THE WRITE PATH ---
        target = random.choice(["film", "payment", "actor", "customer"])
        random_id = random.randint(1, 1000)
        if random.random() > 0.5:
            return f"UPDATE {target} SET last_update = NOW() WHERE {target}_id < {random_id}"
        else:
            return f"DELETE FROM {target} WHERE {target}_id > {random_id}"

    # --- THE READ PATH ---
    # Only if we aren't writing do we start the expensive join-building process.
    depth = random.randint(3, 4)
    start_table = random.choice(list(schema.keys()))
    used_tables = [start_table]

    # Decide if we are aggregating or just selecting
    is_agg = random.random() < 0.3
    query = f"SELECT count(*), avg(random()) " if is_agg else "SELECT * "
    query += f"FROM {start_table}"

    # Join Path (Chain vs Star)
    mode = random.choice(["chain", "star"])
    for _ in range(depth - 1):
        search_tables = [used_tables[-1]] if mode == "chain" else used_tables
        available_joins = []
        for t in search_tables:
            for neighbor, key in schema[t].items():
                if neighbor not in used_tables:
                    available_joins.append((t, neighbor, key))

        if not available_joins: break
        parent, next_t, key = random.choice(available_joins)
        query += f" JOIN {next_t} ON {parent}.{key} = {next_t}.{key}"
        used_tables.append(next_t)

    if is_agg:
        query += f" GROUP BY {start_table}.{start_table}_id"

    if random.random() < 0.20:
        order_table = random.choice(used_tables)
        query += f" ORDER BY {order_table}.{order_table}_id DESC"

    query += f" LIMIT {random.randint(5, 5000)}"
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
            if i % 100 == 0:
                print("Taking a 5-second breath to let the CPU cool...")
                time.sleep(5)
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
    collect(10000)