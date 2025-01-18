import json

# Path to the input and output files
input_file = './Data/filtered_data.jsonl'
output_file = './Data/updated_filtered_data.jsonl'

# Mapping of old labels to new labels
label_mapping = {
    'SUBSIDIE_INFORMATIE': 'SUBSIDIE',
    'HANDLEIDINGEN': 'HANDLEIDING',
    'PROJECT_DETAILS': 'PROJECT',
    'BELEIDSONTWIKKELING': 'BELEID'
}

# Open the input file and the output file
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        record = json.loads(line.strip())  # Parse JSON
        if 'label' in record and record['label'] in label_mapping:
            record['label'] = label_mapping[record['label']]  # Update label
        outfile.write(json.dumps(record) + '\n')  # Write updated record to the output file

print(f"Labels updated and saved to {output_file}")
