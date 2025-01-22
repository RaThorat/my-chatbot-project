# README: Chatbot met Retrieval-Augmented Generation (RAG)

## Inleiding
De Dienst Uitvoering Subsidies aan Instellingen (DUS-I) streeft naar innovatieve oplossingen om interne processen efficiënter te maken en de dienstverlening te verbeteren. Dit project combineert een chatbot met een Retrieval-Augmented Generation (RAG)-architectuur om vragen te beantwoorden op basis van zowel opgehaalde documentfragmenten als gegenereerde antwoorden.

Het doel is een schaalbare, privacyschone oplossing die gebruik maakt van openbare gegevens van DUS-I (zoals beleidsdocumenten en nieuwsberichten) om medewerkers snel en accuraat te informeren.

## Doelstellingen

1. **Demo aan het Bestuur**: Een werkende chatbot tonen die vragen kan beantwoorden met behulp van informatie van DUS-I’s openbare website (bijv. https://www.dus-i.nl/documenten).
2. **Privacybescherming**: Het bieden van mogelijkheid voor het uitvoeren van een Data Protection Impact Assessment (DPIA).
3. **Efficiëntie**: Tijd besparen door snel informatie te leveren aan medewerkers via AI.
4. **Schaalbaarheid**: Het model uitbreiden met nieuwe datasets en functionaliteiten.
5. **Zelfbouw**: Minimaliseren van afhankelijkheid van externe platforms zoals Microsoft Azure of externe ICT-leveranciers.

---

### Use Case Bepaling
- **Identificatie van vragen**: Veelvoorkomende onderwerpen zijn subsidie-informatie, beleidsontwikkelingen en handleidingen.
- **Output**: Korte antwoorden met verwijzingen naar documenten of gegenereerde tekst.

## Architectuur en Aanpak

### Virtual Machine(VM)
- **Specificaties**: Debian GNU/Linux 12, 2 vCPU's, 8.25 GB RAM, 70 GB opslag.
- **Voordelen**: Prodi.gy (annotatie tool) kan lokaal op VM worden gehost en beheerd via SSH.
Bronnen: Prodi.gy Deployment Guide (https://prodi.gy/docs/deployment#vm-deploy) en GCP (google Cloud) Demo (https://www.youtube.com/watch?v=ZLbUtsTgwRM).
- **Kosten**: Kosten voor Google Cloud VM: €200 per maand.

### Dataset Voorbereiding
- **Bronnen**: 46 txt, pdf en odt documenten van de DUS-I website zijn gebruikt om Chunks (200 woorden per chunk) te maken in json formaat. Voor fine tuning van GPT-NL 1.3B zijn de chunks gebruikt. Voor text categorization model, dezelfde documenten omgezet naar JSONL-formaat. Voor named entity recognition model, 30 tekst documenten van de DUS-i website schoongemaakt en gecombineerd. In ./scripts/voorbereiding staan allerlei scripts om de data schoon te maken, in verschillende formaten (txt, json, jsonl) te transformeren.
- **Indexeren**: Chunks geindexeerd met faiss_index.py

### Modelontwikkeling
Tekstclassificatie (Textcat) Model:  Documenten gecategoriseerd (scripts/groeperen_segment_text_to_jsonl.py) in labels zoals:PROJECT, HANDLEIDIG, OVEREENKOMST, PLAN, BELEID, SUBSIDIE. Training uitgevoerd met GroNLP/bert-base-dutch-cased model (110 miljoen parameters); 8vCPU's en 64 GB RAM was vereist. 

python3 ./scripts/train_textcat_model.py

Named Entity Recognition (NER) Model: Annotatie met Prodigy (ner.manual) van entiteiten zoals PERSOON, ORGANISATIE, PROJECT, BEDRAG, LOCATIE, TIJDSPERIODE, SUBSIDIE en PRODUCT zijn geïdentificeerd.

Modeltraining uitgevoerd met de command-line interface van Prodigy:

prodigy ner.manual ner_dataset nl_core_news_lg ./Data/combined_documents.txt --label PERSOON,ORG,PROJECT,BEDRAG,LOC,TIJD,SUB
prodigy train ./models --ner ner_dataset --lang nl --label-stats --verbose --eval-split 0.2

Textcat- en NER-modellen gecombineerd in een pipeline (scripts/ner_textcat_pipeline.py). Response van chatbot is geformuleerd (scripts/dialoogbeheer.py).

#### fine tuning GPT-NL/1.3B model
GPT-NL/1.3B model model is fine tuned als generatieve model met 32vCPU's en 256 GB RAM met fine_tuning_gpt_nl.py

Testen: In scripts/testen staan allerlei tests om de codes en modellen te controleren.

#### Embeddings en Vectorindex
- Documenten omgezet naar embeddings met `all-MiniLM-L6-v2`.
- FAISS gebruikt voor snelle zoekopdrachten in de vectorruimte met faiss_search.py

### Retrieval-Augmented Generation (RAG)
- FAISS opgezet om documenten op te halen op basis van relevantie met faiss_search.py.
- Chatbot integreert in webapp.py:
  1. **Intentieherkenning**: Begrijpen van de vraag via Textcat.
  2. **Entiteitenherkenning**: Uitlezen van entiteiten uit de vraag via NER.
  3. **Documentophaling**: FAISS zoekt relevante documenten. De documenten zijn verder gefilterd op entiteiten.
  4. **Generatief antwoord**: Fine tuned model genereert een antwoord met opgehaalde documenten als context plus intent plus entiteiten.

### Frontend- en Backendintegratie
- **Frontend**: Een interactieve webinterface gebouwd met HTML en JavaScript in ./templates/index.html.
- **Backend**: Flask-server die:
  - Intenties en entiteiten verwerkt.
  - Documenten zoekt in FAISS.
  - Contextuele antwoorden genereert met de fine tuned model.

---

## Hoe te Gebruiken

**Start de Webapp**:
   ```bash
   python3 ./scripts/webapp.py
   ```
   Open de browser op [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## Beperkingen

- **Opslaglimiet**: De models kunnen niet direct worden gehost op GitHub.
- **Generatiekwaliteit**: Fine tuned model kan geen goede antwoorden geven bij simpele vragen (Screenshot from 2025-01-22 15-39-21.png).

---

