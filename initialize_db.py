import sqlite3

# Connect to (or create) the database
conn = sqlite3.connect("resource_allocation.db")
cursor = conn.cursor()

# Create a table for resource allocation if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS allocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    bandwidth_mhz INTEGER,
    power_watts INTEGER,
    qos_requirement TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

# Commit and close the connection
conn.commit()
conn.close()
