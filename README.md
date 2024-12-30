# my-chatbot-project

## Inleiding
De Dienst Uitvoering Subsidies aan Instellingen (DUS-I) streeft naar innovatieve oplossingen om interne processen efficiënter te maken en de dienstverlening aan medewerkers te verbeteren. In lijn met dit streven is een pilot gestart om een chatbot te ontwikkelen die veelgestelde vragen van medewerkers beantwoordt op basis van openbare informatie, zoals beleidsdocumenten en nieuwsberichten. Dit rapport biedt een overzicht van de doelstellingen, aanpak, technische details en verwachte voordelen van deze pilot.

## Doelstellingen
De focus van deze pilot ligt op zelfbouw en minimalisering van afhankelijkheid van externe platforms zoals Microsoft of van ICT leveranciers. De pilot heeft de volgende doelen:

### Demo aan Bestuur: 
Een werkende chatbot (proof of concept) tonen die vragen beantwoordt met behulp van informatie op de website van DUS-I, zoals beleidsstukken en nieuwsartikelen (bijv. https://www.dus-i.nl/documenten).

### Privacybescherming: 
Een Data Protection Impact Assessment (DPIA) mogelijk maken door de ervaring en mening.

### Efficiëntie voor Medewerkers: 
Mening van medewerkers in bouwen van AI meenemen.

### Schaalbaarheid: 
Mogelijkheid controleren tot uitbreiding met nieuwe datasets en functionaliteiten met mogelijkheid tot cloud hosting.

## Aanpak en Stappenplan
### Stap 1: Use Case Bepaling
Identificeren van Vragen: Veelvoorkomende onderwerpen zoals samenvattingen van nieuwsberichten en informatie over subsidieregelingen.

Output Bepalen: Korte antwoorden, informatie of verwijzingen naar relevante documenten.

### Stap 2: Dataset Voorbereiding
Verzameling van Data: 35 openbare documenten van DUS-I (https://www.dus-i.nl/documenten) geselecteerd, zoals handleidingen, bekendmakingen en subsidie-informatie.

Opschoning van Data: Ruis en niet-relevante informatie verwijderd (Data/cleaned_documents.txt). Documenten gecombineerd (scripts/combine_text_files.py) in Data/combined_documents.txt. Voorlopig is geen aandacht besteed aan het voorkomen van gevoelige gegevens; dit wordt meegenomen in de DPIA-evaluatie.

### Stap 3: Gebruik van Virtual Machine
Specificaties: Debian GNU/Linux 12 (bookworm) met 2 vCPU’s, 8.25 GB RAM, en 70 GB opslag.

Voordelen: Prodigy eenvoudig beheren via SSH en VSCode, met lokale functionaliteiten zoals slepen en bewerken van bestanden.

Bronnen: Prodigy Deployment Guide en GCP Demo.

### Stap 4: Modelontwikkeling
Tekstclassificatie (Textcat) Model: Documenten omgezet naar JSONL-formaat (Data/raw_labeled.jsonl). Subdocumenten gecategoriseerd (scripts/groeperen_segment_text_to_jsonl.py) in labels zoals:SUBSIDIE_INFORMATIE, PROJECT_DETAILS, INTERN_DUSI, BELEIDSONTWIKKELING, UITSLAG, HANDLEIDINGEN, INZICHT. Training uitgevoerd met GroNLP/bert-base-dutch-cased model (110 miljoen parameters); 256 GB RAM was vereist. Alternatieve pogingen met GEITje waren niet succesvol.

python3 ./scripts/train_textcat_model.py

Named Entity Recognition (NER) Model: Annotatie met Prodigy (ner.manual) van entiteiten zoals PERSOON, ORGANISATIE, PROJECT, BEDRAG, LOCATIE, TIJDSPERIODE, SUBSIDIE en PRODUCT zijn geïdentificeerd.

Modeltraining uitgevoerd met de command-line interface van Prodigy:

prodigy ner.manual ner_dataset nl_core_news_lg ./Data/combined_documents.txt --label PERSOON,ORG,PROJECT,BEDRAG,LOC,TIJD,SUB
prodigy train ./models --ner ner_dataset --lang nl --label-stats --verbose --eval-split 0.2

Textcat- en NER-modellen gecombineerd in een pipeline (scripts/ner_textcat_pipeline.py). Response van chatbot is geformuleerd (scripts/dialoogbeheer.py).

Testen: scripts/test_pipeline.py, scripts/test_spacy_ner.py,

### Stap 5: Integreer de frontend met de backend


Frontend-code (chatbot.html) is gemaakt en geplaatst in /my-chatbot-project/frontend/chatbot.html.
Gebruik Python om een lokale webserver te starten voor html pagina:
python3 -m http.server 8000 --directory ./frontend
Ga naar http://127.0.0.1:8000/chatbot.html om de chatbot te gebruiken.

Backend ontwikkeld met Flask (main.py).

## Beperkingen:
Grote bestanden zoals models/textcat_model/model.safetensors konden niet op GitHub worden gehost vanwege opslaglimieten

## Kostenindicatie:
Kosten voor Google Cloud Console: €100 voor RAM-gebruik tijdens modeltraining.