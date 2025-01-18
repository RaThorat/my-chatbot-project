import os

# Function to convert text content to Markdown
def convert_to_markdown(content):
    """
    Converts text/tables into Markdown format.
    """
    lines = content.splitlines()
    markdown_output = []

    if lines and "," in lines[0]:  # Detect CSV/table format
        markdown_output.append("| " + " | ".join(lines[0].split(",")) + " |")
        markdown_output.append("|" + "---|" * len(lines[0].split(",")))
        for line in lines[1:]:
            markdown_output.append("| " + " | ".join(line.split(",")) + " |")
    else:
        for line in lines:
            if line.strip():
                # Apply `###` only to lines that look like headings
                if line.strip().endswith(":") or line.strip()[0].isupper():
                    markdown_output.append(f"### {line.strip()}")
                else:
                    markdown_output.append(line.strip())

    return "\n".join(markdown_output)

# Function to save content to a file
def save_to_file(output_path, content):
    with open(output_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(content)

# Function to normalize text files and convert them to Markdown
def normalize_txt_file(input_path, output_path):
    try:
        with open(input_path, "r", encoding="utf-8") as infile:
            text = infile.read()

        # Check if already in Markdown format
        if text.lstrip().startswith("#") or "|" in text:
            save_to_file(output_path, text)  # Save as is
            print(f"File already in Markdown format: {output_path}")
            return

        # Otherwise, convert to Markdown
        markdown_content = convert_to_markdown(text)
        save_to_file(output_path, markdown_content)
        print(f"Text file normalized and converted to Markdown: {output_path}")
    except Exception as e:
        print(f"Error normalizing file: {e}")

# Main function to process all .txt files in a directory
def process_files(input_folder, temp_folder):
    os.makedirs(temp_folder, exist_ok=True)
    processed_count = 0
    skipped_count = 0

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        temp_path = os.path.join(temp_folder, os.path.splitext(filename)[0] + ".txt")

        try:
            if filename.endswith(".txt"):
                normalize_txt_file(input_path, temp_path)
                processed_count += 1
            else:
                print(f"Skipping non-text file: {filename}")
                skipped_count += 1
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            skipped_count += 1

    print(f"Processing complete. {processed_count} text files processed, {skipped_count} files skipped.")

# Paths
input_folder = "./Data/raw"
temp_folder = "./Data/processed"

# Process files
process_files(input_folder, temp_folder)
