import sqlite3
import pandas as pd

# Connect to the database
print("Connecting to database...")
conn = sqlite3.connect('instance/database.db')

# Get a list of all tables
print("\nTables in the database:")
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
print(tables)

# Show contents of each table
for table in tables['name']:
    print(f"\nContents of table: {table}")
    print("=" * 50)
    try:
        data = pd.read_sql(f"SELECT * FROM {table} LIMIT 10;", conn)
        print(data)
    except Exception as e:
        print(f"Could not read table {table}: {str(e)}")
    print("\n" + "="*50 + "\n")

conn.close()
print("\nDatabase connection closed.")
