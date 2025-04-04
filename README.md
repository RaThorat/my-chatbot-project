# Chatbot met Retrieval-Augmented Generation (RAG)

## Inleiding

De Dienst Uitvoering Subsidies aan Instellingen (DUS-I) streeft naar innovatieve oplossingen om interne processen efficiënter te maken en de dienstverlening te verbeteren. Dit project combineert een chatbot met een Retrieval-Augmented Generation (RAG)-architectuur om vragen te beantwoorden op basis van zowel opgehaalde documentfragmenten als gegenereerde antwoorden.

Het doel is een schaalbare, privacyschone oplossing die gebruik maakt van openbare gegevens van DUS-I (zoals beleidsdocumenten en nieuwsberichten) om medewerkers snel en accuraat te informeren.

## Doelstellingen

- **Demo aan het Bestuur:** Een werkende chatbot tonen die vragen kan beantwoorden met behulp van informatie van DUS-I’s openbare website (bijv. [DUS-I Documenten](https://www.dus-i.nl/documenten)).
- **Privacybescherming:** Het bieden van mogelijkheid voor het uitvoeren van een Data Protection Impact Assessment (DPIA).
- **Efficiëntie:** Tijd besparen door snel informatie te leveren aan medewerkers via AI.
- **Schaalbaarheid:** Het model uitbreiden met nieuwe datasets en functionaliteiten.
- **Zelfbouw:** Minimaliseren van afhankelijkheid van externe platforms zoals Microsoft Azure of externe ICT-leveranciers.

## Use Case Bepaling

- **Identificatie van vragen:** Veelvoorkomende onderwerpen zijn subsidie-informatie, beleidsontwikkelingen en handleidingen.
- **Output:** Korte antwoorden met verwijzingen naar documenten of gegenereerde tekst.

## Architectuur en Aanpak

### Virtual Machine (VM)

- **Specificaties:** Debian GNU/Linux 12, 2 vCPU's, 8.25 GB RAM, 70 GB opslag.
- **Voordelen:** Prodi.gy (annotatie tool) kan lokaal op VM worden gehost en beheerd via SSH.
- **Bronnen:**
  - [Prodi.gy Deployment Guide](https://prodi.gy/docs/deployment#vm-deploy)
  - [GCP Demo](https://www.youtube.com/watch?v=ZLbUtsTgwRM)
- **Kosten:** Kosten voor Google Cloud VM: €200 per maand.

### Dataset Voorbereiding

- **Bronnen:** 46 txt, pdf en odt documenten van de DUS-I website zijn gebruikt om Chunks (200 woorden per chunk) te maken in JSON-formaat.
- **Fine tuning:**
  - Voor fine tuning van GPT-NL 1.3B zijn de chunks gebruikt.
  - Voor text categorization model: dezelfde documenten omgezet naar JSONL-formaat.
  - Voor named entity recognition model: 30 tekst documenten van de DUS-I website schoongemaakt en gecombineerd.
- **Scripts:** In `./scripts/voorbereiding` staan allerlei scripts om de data schoon te maken en om te zetten in verschillende formaten (txt, json, jsonl).
- **Indexeren:** Chunks geïndexeerd met `faiss_index.py`.

## Modelontwikkeling

### Tekstclassificatie (Textcat) Model

- **Categorisatie:** Documenten gegroepeerd (`groeperen_segment_text_to_jsonl.py`) in labels zoals: `PROJECT`, `HANDLEIDING`, `OVEREENKOMST`, `PLAN`, `BELEID`, `SUBSIDIE`.
- **Training:**
  - Uitgevoerd met `GroNLP/bert-base-dutch-cased` model (110 miljoen parameters).
  - 8 vCPU's en 64 GB RAM was vereist.
  - Script voor textcat model: `train_textcat_model.py`.

### Named Entity Recognition (NER) Model

- **Annotatie met Prodigy (`ner.manual`)** van entiteiten zoals:
  - `PERSOON`, `ORGANISATIE`, `PROJECT`, `BEDRAG`, `LOCATIE`, `TIJDSPERIODE`, `SUBSIDIE`, `PRODUCT`.

- **Training uitgevoerd met:**

  ```sh
  prodigy ner.manual ner_dataset nl_core_news_lg ./Data/combined_documents.txt --label PERSOON,ORG,PROJECT,BEDRAG,LOC,TIJD,SUB
  
  prodigy train ./models --ner ner_dataset --lang nl --label-stats --verbose --eval-split 0.2
  ```

- **Integratie:** Textcat- en NER-modellen gecombineerd in een pipeline (`ner_textcat_pipeline.py`). De ner_dataset.jsonl in map Data is de geannoteerde data van de tekst combined_text.txt.

### Fine Tuning GPT-NL/1.3B model

- **Fine tuning:** GPT-NL/1.3B model getraind als generatief model met `fine_tuning_gpt_nl.py`.
- **Benodigdheden:** 32 vCPU's en 256 GB RAM.
- **Testen:** In `scripts/testen` staan tests om de codes en modellen te controleren.

## Embeddings en Vectorindex

- **Embeddings:** Documenten omgezet naar embeddings met `all-MiniLM-L6-v2`.
- **FAISS:** Gebruikt voor snelle zoekopdrachten in de vectorruimte (`faiss_search.py`).

## Retrieval-Augmented Generation (RAG)

- **FAISS-opzet:** Documenten opgehaald op basis van relevantie (`faiss_search.py`).
- **Chatbot-integratie (`webapp.py`):**
  - **Intentieherkenning:** Begrijpen van de vraag via Textcat.
  - **Entiteitenherkenning:** Uitlezen van entiteiten uit de vraag via NER.
  - **Documentophaling:** FAISS zoekt relevante documenten en filtert op entiteiten.
  - **Generatief antwoord:** Fine tuned model genereert een antwoord met opgehaalde documenten als context plus intent plus entiteiten.

## Frontend- en Backendintegratie

- **Frontend:** Een interactieve webinterface gebouwd met HTML en JavaScript (`./templates/index.html`).
- **Backend:** Flask-server die:
  - Intenties en entiteiten verwerkt.
  - Documenten zoekt in FAISS.
  - Contextuele antwoorden genereert met het fine tuned model.

## Hoe te Gebruiken

Start de webapplicatie:

```sh
python3 ./scripts/webapp.py
```

Open de browser op `http://127.0.0.1:5000`.

## Beperkingen

- **Opslaglimiet:** De modellen kunnen niet direct worden gehost op GitHub.
- **Generatiekwaliteit:** Fine tuned model kan geen goede antwoorden geven bij simpele vragen (`Screenshot from 2025-01-22 15-39-21.png`).

