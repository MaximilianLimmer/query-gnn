import psycopg2
import json
import random
import statistics


def get_connection():
    return psycopg2.connect(dbname="dvdrental", user="postgres", host="127.0.0.1")

SCHEMA_CONNECTIONS = {
    frozenset(["actor", "film_actor"]): "actor_id",
    frozenset(["film", "film_actor"]): "film_id",
    frozenset(["film", "film_category"]): "film_id",
    frozenset(["category", "film_category"]): "category_id",
    frozenset(["film", "inventory"]): "film_id",
    frozenset(["inventory", "rental"]): "inventory_id",
    frozenset(["rental", "customer"]): "customer_id",
    frozenset(["rental", "staff"]): "staff_id",
    frozenset(["rental", "payment"]): "rental_id",
    frozenset(["customer", "payment"]): "customer_id",
    frozenset(["customer", "address"]): "address_id",
    frozenset(["staff", "address"]): "address_id",
    frozenset(["store", "address"]): "address_id",
    frozenset(["address", "city"]): "city_id",
    frozenset(["city", "country"]): "country_id"
}

# Adjacency list to guide the "walk" through the database
FULL_SCHEMA_GRAPH = {
    "actor": ["film_actor"],
    "film_actor": ["actor", "film"],
    "film": ["film_actor", "film_category", "inventory"],
    "film_category": ["film", "category"],
    "category": ["film_category"],
    "inventory": ["film", "rental"],
    "customer": ["address", "rental", "payment"],
    "rental": ["inventory", "customer", "staff", "payment"],
    "payment": ["customer", "staff", "rental"],
    "staff": ["address", "rental", "payment"],
    "address": ["city", "customer", "staff", "store"],
    "city": ["address", "country"],
    "country": ["city"],
    "store": ["address"]
}


def generate_random_query():
    """Generates a valid JOIN query based on the schema graph."""
    depth = random.randint(1, 4)
    start_table = random.choice(list(FULL_SCHEMA_GRAPH.keys()))
    used_tables = [start_table]

    # We use JOIN ... USING (key) to avoid duplicate column errors with SELECT *
    query_body = f"FROM {start_table}"

    current_table = start_table
    for _ in range(depth):
        # Find neighbors we haven't visited yet
        options = [t for t in FULL_SCHEMA_GRAPH[current_table] if t not in used_tables]
        if not options:
            break

        next_table = random.choice(options)
        key = SCHEMA_CONNECTIONS.get(frozenset([current_table, next_table]))

        if key:
            query_body += f" JOIN {next_table} USING ({key})"
            used_tables.append(next_table)
            current_table = next_table
        else:
            break

    # Construct final SQL
    # Using SELECT * for simplicity, or random columns for complexity
    sql = f"SELECT * {query_body}"

    # Add a random filter to vary the row counts (cardinality)
    if random.random() < 0.7:
        sql += f" WHERE random() < {random.uniform(0.01, 0.9)}"

    sql += f" LIMIT {random.randint(10, 1000)}"
    return sql

def collect_dataset(size=20000, output_file="query_data.json"):
    dataset = []
    conn = None

    try:
        conn = get_connection()
        cur = conn.cursor()
        print(f"Collecting {size} queries")

        for i in range(1, size + 1):
            sql = generate_random_query()
            runtimes = []
            plans = []

            try:
                # Run 3 times to get a stable median (handles cache warming)
                for _ in range(3):
                    cur.execute("BEGIN;")
                    # EXPLAIN ANALYZE gives us the real ground-truth time
                    cur.execute(f"EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS) {sql}")
                    raw_plan = cur.fetchone()[0][0]['Plan']
                    runtimes.append(raw_plan['Actual Total Time'])
                    plans.append(raw_plan)
                    cur.execute("ROLLBACK;")

                median_time = statistics.median(runtimes)
                best_plan_idx = runtimes.index(median_time)

                dataset.append({
                    "sql": sql,
                    "plan": plans[best_plan_idx],
                    "runtime": median_time
                })

            except Exception as e:
                # If a query fails (e.g. timeout or syntax), rollback and skip
                if conn: conn.rollback()
                continue

            if i % 100 == 0:
                print(f"Progress: {i}/{size} | Last Median: {median_time:.2f}ms")

        # Save results
        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=2)

        print(f"\nSaved {len(dataset)} valid samples to {output_file}")

    except Exception as e:
        print(f"Fatal Error: {e}")
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    # Start small to test, then bump to 20000
    collect_dataset(size=5000)