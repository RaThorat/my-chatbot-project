from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import psutil

# Controleer CPU-configuratie
print(f"CPU cores in gebruik: {psutil.cpu_count(logical=False)}")
print(f"Totaal CPU cores: {psutil.cpu_count(logical=True)}")

# Laad een kleiner GPT-Neo-model
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Laad de dataset
dataset = load_dataset(
    "json",
    data_files={"train": "./Data/train.json", "validation": "./Data/validation.json"},
    field="Documents"
)

# Tokenizer-functie aanpassen voor het JSON-formaat
def tokenize_function(example):
    try:
        chunks = [chunk["Chunk"] for chunk in example["Chunks"]]
        text = " ".join(chunks)
        return tokenizer(text, truncation=True, padding="max_length", max_length=512)
    except KeyError as e:
        print(f"KeyError: {e} in {example}")
        return {"input_ids": [], "attention_mask": []}

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["Chunks"])

# Stel PyTorch in voor CPU-gebruik
torch.set_num_threads(4)

# Trainingconfiguratie
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=2,
    dataloader_num_workers=4,
    logging_dir="./logs"
)

# Trainer instellen
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Start de training
trainer.train()

# Sla het model en de tokenizer op
model.save_pretrained("./models/fine_tuned_generative_model")
tokenizer.save_pretrained("./models/fine_tuned_generative_model")
