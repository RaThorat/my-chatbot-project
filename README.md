# README: Chatbot met Retrieval-Augmented Generation (RAG)

## Inleiding
De Dienst Uitvoering Subsidies aan Instellingen (DUS-I) streeft naar innovatieve oplossingen om interne processen efficiënter te maken en de dienstverlening te verbeteren. Dit project combineert een chatbot met een Retrieval-Augmented Generation (RAG)-architectuur om vragen te beantwoorden op basis van zowel opgehaalde documentfragmenten als gegenereerde antwoorden.

Het doel is een schaalbare, privacyschone oplossing die gebruik maakt van openbare gegevens van DUS-I (zoals beleidsdocumenten en nieuwsberichten) om medewerkers snel en accuraat te informeren.

## Doelstellingen

1. **Demo aan het Bestuur**: Een werkende chatbot tonen die vragen kan beantwoorden met behulp van informatie van DUS-I’s openbare website (bijv. https://www.dus-i.nl/documenten).
2. **Privacybescherming**: Het waarborgen van gegevensbescherming door het uitvoeren van een Data Protection Impact Assessment (DPIA).
3. **Efficiëntie**: Tijd besparen door snel informatie te leveren aan medewerkers via AI.
4. **Schaalbaarheid**: Het model uitbreiden met nieuwe datasets en functionaliteiten.
5. **Zelfbouw**: Minimaliseren van afhankelijkheid van externe platforms zoals Microsoft of ICT-leveranciers.

---

## Architectuur en Aanpak

### Stap 1: Use Case Bepaling
- **Identificatie van vragen**: Veelvoorkomende onderwerpen zijn subsidie-informatie, beleidsontwikkelingen en handleidingen.
- **Output**: Korte antwoorden met verwijzingen naar documenten of gegenereerde tekst.

### Stap 2: Dataset Voorbereiding
- **Bronnen**: 35 openbare documenten verzameld van de DUS-I-website.
- **Opschoning**: Documenten gecombineerd en ruis verwijderd met scripts (`combine_text_files.py`).
- **Indexeren**: Gecombineerde documenten (`Data/combined_documents.txt`) zijn verwerkt en opgeslagen in een SQLite-database (`documents.db`).

### Stap 3: Virtual Machine
- **Specificaties**: Debian GNU/Linux 12, 2 vCPU's, 8.25 GB RAM, 70 GB opslag.
- **Voordelen**: Prodigy en andere tools kunnen lokaal worden gehost en beheerd via SSH.
Bronnen: Prodigy Deployment Guide en GCP Demo.

### Stap 4: Modelontwikkeling
Tekstclassificatie (Textcat) Model: Documenten omgezet naar JSONL-formaat (Data/raw_labeled.jsonl). Subdocumenten gecategoriseerd (scripts/groeperen_segment_text_to_jsonl.py) in labels zoals:SUBSIDIE_INFORMATIE, PROJECT_DETAILS, INTERN_DUSI, BELEIDSONTWIKKELING, UITSLAG, HANDLEIDINGEN, INZICHT. Training uitgevoerd met GroNLP/bert-base-dutch-cased model (110 miljoen parameters); 128 GB RAM was vereist. Alternatieve pogingen met GEITje waren niet succesvol.

python3 ./scripts/train_textcat_model.py

Named Entity Recognition (NER) Model: Annotatie met Prodigy (ner.manual) van entiteiten zoals PERSOON, ORGANISATIE, PROJECT, BEDRAG, LOCATIE, TIJDSPERIODE, SUBSIDIE en PRODUCT zijn geïdentificeerd.

Modeltraining uitgevoerd met de command-line interface van Prodigy:

prodigy ner.manual ner_dataset nl_core_news_lg ./Data/combined_documents.txt --label PERSOON,ORG,PROJECT,BEDRAG,LOC,TIJD,SUB
prodigy train ./models --ner ner_dataset --lang nl --label-stats --verbose --eval-split 0.2

Textcat- en NER-modellen gecombineerd in een pipeline (scripts/ner_textcat_pipeline.py). Response van chatbot is geformuleerd (scripts/dialoogbeheer.py).

Testen: scripts/test_pipeline.py, scripts/test_spacy_ner.py,

#### Embeddings en Vectorindex
- Documenten omgezet naar embeddings met `all-MiniLM-L6-v2`.
- FAISS gebruikt voor snelle zoekopdrachten in de vectorruimte.

### Stap 5: Retrieval-Augmented Generation (RAG)
- FAISS opgezet om documenten op te halen op basis van relevantie.
- Chatbot integreert:
  1. **Intentieherkenning**: Begrijpen van de vraag via Textcat.
  2. **Entiteitenherkenning**: Uitlezen van sleutelbegrippen via NER.
  3. **Documentophaling**: FAISS zoekt relevante documenten.
  4. **Generatief antwoord**: GPT-Neo 125M genereert een antwoord met opgehaalde documenten als context.

### Stap 6: Frontend- en Backendintegratie
- **Frontend**: Een interactieve webinterface gebouwd met HTML en JavaScript.
- **Backend**: Flask-server die:
  - Intenties en entiteiten verwerkt.
  - Documenten zoekt in SQLite en FAISS.
  - Contextuele antwoorden genereert met GPT-Neo 125M.

---

## Hoe te Gebruiken

1. **Installatie**:
   - Clone de repository.
   - Installeer vereisten: `pip install -r requirements.txt`.

2. **Data Voorbereiden**:
   - Plaats ruwe documenten in `./Data/raw`.
   - Combineer en indexeer met de scripts in `./scripts`.

3. **FAISS Index Bouwen**:
   ```bash
   python3 ./scripts/faiss_index.py
   ```

4. **Start de Webapp**:
   ```bash
   python3 ./scripts/webapp.py
   ```
   Open de browser op [http://127.0.0.1:5000](http://127.0.0.1:5000).

5. **Test de Pipeline**:
   ```bash
   python3 ./scripts/test_pipeline.py
   ```

---

## Beperkingen

- **Opslaglimiet**: Grote modellen zoals `textcat_model` kunnen niet direct worden gehost op GitHub.
- **Generatiekwaliteit**: GPT-Neo 125M kan beperkte antwoorden geven bij complexe vragen.

---

## Kosten
- Kosten voor Google Cloud VM: €156 per maand.

---

## Toekomstige Uitbreidingen
- Integreren van geavanceerdere modellen zoals GPT-3 voor betere antwoordkwaliteit.
- Ondersteuning voor meerdere talen.
- Hosting op schaalbare cloudomgevingen.
