import sqlite3
db_path = "documents.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
query = "dus-i"
cursor.execute("SELECT filename, content FROM docs WHERE docs MATCH ?", (f'"{query}"',))
results = cursor.fetchall()
print(results)
conn.close()
