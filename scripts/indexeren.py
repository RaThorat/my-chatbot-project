import os
import re
import sqlite3

# Path to the SQLite database
db_path = "documents.db"

# Function to clean document content
def clean_text(content):
    # Remove HTML tags
    content = re.sub(r"<.*?>", "", content)
    # Remove excessive whitespace
    content = re.sub(r"\s+", " ", content).strip()
    return content

# Function to load documents from a directory
def load_documents(data_directory):
    docs = []
    for filename in os.listdir(data_directory):
        file_path = os.path.join(data_directory, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    cleaned_content = clean_text(content)
                    docs.append((filename, cleaned_content))  # Store filename and content
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    return docs

# Function to create the SQLite database and table
def create_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create a table with full-text search enabled
    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS docs USING FTS5(
        filename,
        content
    )
""")

    conn.commit()
    conn.close()

# Function to index documents in SQLite with duplicate check
def index_documents(db_path, documents):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for doc in documents:
        try:
            # Check if the document already exists based on filename
            cursor.execute("SELECT COUNT(*) FROM docs WHERE filename = ?", (doc[0],))
            exists = cursor.fetchone()[0]
            
            if exists:
                print(f"Document '{doc[0]}' already exists. Skipping.")
            else:
                cursor.execute("INSERT INTO docs (filename, content) VALUES (?, ?)", doc)
        except Exception as e:
            print(f"Error indexing document {doc[0]}: {e}")
    
    conn.commit()
    conn.close()

# Function to search the SQLite index
def search_documents(db_path, query):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Wrap the query in quotes to handle spaces and special characters
    formatted_query = f'"{query}"'
    cursor.execute("SELECT filename, content FROM docs WHERE docs MATCH ?", (formatted_query,))
    results = cursor.fetchall()
    
    conn.close()
    return results

# Directory containing raw documents
data_directory = "./Data/processed"

# Main script
if __name__ == "__main__":
    # Step 1: Load documents
    documents = load_documents(data_directory)
    print(f"Loaded {len(documents)} documents.")

    # Step 2: Create SQLite database
    create_database(db_path)

    # Step 3: Index documents
    index_documents(db_path, documents)
    print(f"Indexed {len(documents)} documents.")

    # Step 4: Test search
    query = "DUS-i"  # Replace with your search term
    results = search_documents(db_path, query)
    
    print(f"Found {len(results)} results for query '{query}':")
    for filename, content in results:
        print(f"- {filename}: {content[:100]}...")  # Print first 100 characters
