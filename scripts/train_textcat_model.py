from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torchvision
torchvision.disable_beta_transforms_warning()

# 1. Laad de Data
dataset = load_dataset("json", data_files={"train": "/home/RaThorat/my-chatbot-project/Data/doc_chunks.jsonl"})
label_list = list(set([item["label"] for item in dataset["train"]]))  # Automatisch labels ophalen

# 2. Preprocessing
model_name = "GroNLP/bert-base-dutch-cased"  # 
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 3. Bereid Labels voor
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

def encode_labels(examples):
    examples["label"] = label2id[examples["label"]]
    return examples

tokenized_datasets = tokenized_datasets.map(encode_labels)

# **NEW SECTION: Split into Train and Eval Datasets**
# 4. Splits dataset into train and eval
split_dataset = tokenized_datasets["train"].train_test_split(test_size=0.2)  # Split 20% for eval
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# 5. Model Configureren
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# 6. Training Instellen
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Ensure eval_dataset is provided
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10
)

# 7. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 8. Trainer Initialiseren
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Use the split train dataset
    eval_dataset=eval_dataset,    # Use the split eval dataset
    tokenizer=tokenizer,          # Warning for deprecation can be ignored for now
    compute_metrics=compute_metrics
)

# 9. Train het Model
trainer.train()

# 10. Sla het Model op
trainer.save_model("./models/textcat_model")
tokenizer.save_pretrained("./models/textcat_model")
