import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import split_sentences
from prodigy.util import set_hashes
from transformers import AutoTokenizer, pipeline

@prodigy.recipe(
    "ner.manual_hf",
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=("Path to the input file", "positional", None, str),
    labels=("Comma-separated labels", "option", "l", str),
)
def ner_manual_hf(dataset, source, labels=None):
    """Custom recipe to use Hugging Face model for NER in Prodigy."""
    # Load the tokenizer and model
    model_name = "pdelobelle/bert-base-dutch-cased-finetuned-ner"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp = pipeline("ner", model=model_name, tokenizer=tokenizer, aggregation_strategy="simple")

    # Load and preprocess the input data
    with open(source, "r", encoding="utf-8") as f:
        raw_text = f.readlines()
    stream = [{"text": line.strip()} for line in raw_text if line.strip()]

    # Map Hugging Face model labels to your custom labels
    label_mapping = {
        "LABEL_0": "PERSOON",
        "LABEL_1": "ORGANISATIE",
        "LABEL_2": "PROJECT",
        "LABEL_3": "BEDRAG",
        "LABEL_4": "LOCATIE",
        "LABEL_5": "TIJDSPERIODE",
        "LABEL_6": "SUBSIDIE"
    }

    def add_annotations(stream):
        for task in stream:
            # Add unique hashes to tasks
            task = set_hashes(task)
            text = task["text"]

            # Tokenize the text
            tokens = tokenizer(text, return_offsets_mapping=True, truncation=True)
            token_offsets = tokens["offset_mapping"]
            token_list = [{"text": text[start:end], "start": start, "end": end, "id": i}
                          for i, (start, end) in enumerate(token_offsets) if start != end]

            # Get NER results
            ner_results = nlp(text)
            spans = []
            for ent in ner_results:
                # Map entity spans to tokens
                token_start = next((i for i, t in enumerate(token_list) if t["start"] == ent["start"]), None)
                token_end = next((i for i, t in enumerate(token_list) if t["end"] == ent["end"]), None)
                if token_start is not None and token_end is not None:
                    spans.append({
                        "start": ent["start"],
                        "end": ent["end"],
                        "token_start": token_start,
                        "token_end": token_end,
                        "label": label_mapping.get(ent["entity_group"], ent["entity_group"])  # Map label
                    })

            # Add tokens and spans to the task
            task["tokens"] = token_list
            task["spans"] = spans
            yield task

    # Annotate and process the stream
    annotated_stream = add_annotations(stream)

    # Exclude already-annotated examples
    def exclude_annotated_tasks(stream, dataset):
        from prodigy.components.db import connect
        db = connect()
        annotated_hashes = {eg["_input_hash"] for eg in db.get_dataset(dataset)}
        for task in stream:
            if task["_input_hash"] not in annotated_hashes:
                yield task

    annotated_stream = exclude_annotated_tasks(annotated_stream, dataset)

    # Return the recipe components
    return {
        "dataset": dataset,
        "view_id": "ner_manual",
        "stream": annotated_stream,
        "config": {
            "labels": labels.split(",") if labels else [],
            "exclude_by": "input",
        },
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python ner_custom.py <dataset> <source_file> <labels>")
        sys.exit(1)

    dataset = sys.argv[1]
    source_file = sys.argv[2]
    labels = sys.argv[3]

    prodigy.serve(
        "ner.manual_hf",
        dataset,
        source_file,
        labels
    )

