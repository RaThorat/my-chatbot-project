import sqlite3

db_path = "documents.db"

try:
    conn = sqlite3.connect(db_path)
    print("Connected to database successfully.")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM docs")
    print(f"Number of documents in database: {cursor.fetchone()[0]}")
    conn.close()
except Exception as e:
    print(f"Error connecting to database: {e}")
