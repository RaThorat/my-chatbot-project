import os
import sqlite3

def create_database(db_path):
    """
    Creates the documents.db database with the appropriate schema.
    """
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create the 'docs' table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS docs (
                rowid INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                markdown TEXT DEFAULT NULL
            )
        ''')

        print(f"Database created successfully at {db_path}.")
    except sqlite3.Error as e:
        print(f"Error creating database: {e}")
    finally:
        conn.close()

def load_data_from_directory(input_folder):
    """
    Loads text data from a directory of files.
    """
    documents = []
    for filename in sorted(os.listdir(input_folder)):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read().strip()
                    documents.append((content, None))  # Markdown will be None for now
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    return documents

def insert_data_into_db(db_path, data):
    """
    Inserts data into the 'docs' table.
    """
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Insert data
        cursor.executemany('INSERT INTO docs (content, markdown) VALUES (?, ?)', data)
        conn.commit()

        print(f"{len(data)} records inserted successfully.")
    except sqlite3.Error as e:
        print(f"Error inserting data: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    # Paths
    db_path = 'documents.db'
    
    processed_data_folder = './Data/processed'

    # Create the database
    create_database(db_path)

    # Load processed data
    
    print("Loading processed data...")
    processed_data = load_data_from_directory(processed_data_folder)
    print(f"Loaded {len(processed_data)} documents from processed data.")

    # Insert data into the database
    insert_data_into_db(db_path, processed_data)

    print("Database setup complete.")
