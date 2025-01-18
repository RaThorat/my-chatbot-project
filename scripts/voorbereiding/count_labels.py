import json
from collections import Counter
import pandas as pd

# Path to the JSONL file
file_path = './Data/filtered_data.jsonl'

# Initialize an empty list to store labels
labels = []

# Read the JSONL file and extract labels
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        record = json.loads(line.strip())
        if 'label' in record:
            labels.append(record['label'])

# Count frequencies of labels
label_counts = Counter(labels)

# Convert to DataFrame for better visualization
df = pd.DataFrame(label_counts.items(), columns=["Label", "Frequency"])

# Display the DataFrame
print(df)

# Optionally, save the output to a CSV file
output_path = './Data/label_frequencies.csv'
df.to_csv(output_path, index=False)
print(f"Label frequencies saved to {output_path}")
