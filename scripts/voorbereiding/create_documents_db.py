import os
import sqlite3
import logging

# Set up logging
logging.basicConfig(
    filename="database_population.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def create_database(db_path):
    """
    Creates the documents.db database with the appropriate schema.
    """
    try:
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
        logging.info(f"Database created successfully at {db_path}.")
    except sqlite3.Error as e:
        logging.error(f"Error creating database: {e}")
    finally:
        conn.close()

def load_data_from_directory(input_folder):
    """
    Loads Markdown files from a directory into memory.
    """
    documents = []
    skipped_files = []

    for filename in sorted(os.listdir(input_folder)):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read().strip()  # Content is already Markdown
                    if content:
                        documents.append((content, content))  # Both `content` and `markdown` are the same
                    else:
                        skipped_files.append(filename)
            except Exception as e:
                logging.error(f"Error reading file {filename}: {e}")
    return documents


def insert_data_into_db(db_path, data):
    """
    Inserts data into the 'docs' table.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.executemany('INSERT INTO docs (content, markdown) VALUES (?, ?)', data)
        conn.commit()

        logging.info(f"Inserted {len(data)} records into the database.")
    except sqlite3.Error as e:
        logging.error(f"Error inserting data into database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    db_path = 'documents.db'
    processed_data_folder = './Data/processed'

    # Step 1: Create the database
    create_database(db_path)

    # Step 2: Load processed data
    logging.info("Loading processed data...")
    processed_data = load_data_from_directory(processed_data_folder)
    logging.info(f"Total processed files: {len(processed_data)}")

    # Step 3: Insert data into the database
    if processed_data:
        insert_data_into_db(db_path, processed_data)
        logging.info("Database setup complete.")
    else:
        logging.warning("No processed data found to insert into the database.")
