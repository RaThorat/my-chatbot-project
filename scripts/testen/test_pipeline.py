import sys
import os

# Voeg de scripts-map toe aan het Python-pad
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ner_textcat_pipeline import process_user_input


result = process_user_input("Wat doet DUS-i?")

print(result)

