import os

def combine_text_files(input_folder, output_file):
    """
    Combineer alle .txt-bestanden in een map tot één bestand.

    Parameters:
    - input_folder (str): Pad naar de map met .txt-bestanden.
    - output_file (str): Pad naar het bestand waarin de samengevoegde teksten worden opgeslagen.
    """
    # Controleer of de map bestaat
    if not os.path.exists(input_folder):
        print(f"De map '{input_folder}' bestaat niet.")
        return

    # Open het uitvoerbestand
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Doorloop alle bestanden in de map
        for filename in sorted(os.listdir(input_folder)):
            if filename.endswith(".txt"):
                file_path = os.path.join(input_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    # Schrijf de inhoud van het bestand naar het uitvoerbestand
                    outfile.write(f"### Bestandsnaam: {filename} ###\n")
                    outfile.write(infile.read())
                    outfile.write("\n\n")  # Voeg extra witregels toe tussen bestanden
        print(f"Alle bestanden zijn samengevoegd in '{output_file}'.")

# Pad naar de map met .txt-bestanden
input_folder = "/home/gebruiker/Documenten/git_workspace/my-chatbot-project/Data/raw"

# Pad naar het uitvoerbestand
output_file = "combined_documents.txt"

# Voer de functie uit
combine_text_files(input_folder, output_file)

