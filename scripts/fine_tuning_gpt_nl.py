from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import psutil
import os

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Display CPU and memory configuration
print(f"CPU cores available: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
print(f"Using 32 vCPUs and 256 GB memory for fine-tuning.")

# Load GPT-Neo 1.3B model (Dutch language version)
model_name = "yhavinga/gpt-neo-1.3B-dutch"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure pad_token is defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
dataset = load_dataset(
    "json",
    data_files={"train": "./Data/train.json", "validation": "./Data/validation.json"},
    field="Documents"
)

# Tokenization function
def tokenize_function(examples):
    # Join all "Chunk" text within each document's Chunks list
    try:
        texts = [
            " ".join(chunk["Chunk"] for chunk in doc) for doc in examples["Chunks"]
        ]
    except (KeyError, TypeError) as e:
        print(f"Error processing examples['Chunks']: {e}. Check the dataset structure.")
        raise

    # Tokenize the joined texts
    tokenized_output = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output


# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["Chunks"])

# Configure PyTorch for CPU usage
torch.set_num_threads(32)  # Fully utilize your 32 vCPUs

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",               # Evaluate after each epoch
    save_strategy="epoch",                     # Save checkpoints after each epoch
    learning_rate=3e-5,                        # Lower learning rate for stability
    per_device_train_batch_size=2,             # Reduced batch size for CPU constraints
    gradient_accumulation_steps=16,            # Accumulate gradients to simulate larger effective batch size
    num_train_epochs=3,                        # Number of epochs
    dataloader_num_workers=8,                  # Parallelize data loading
    logging_dir="./logs",                      # Logging directory
    save_total_limit=2,                        # Limit the number of checkpoints saved
    logging_steps=50,                          # Log every 50 steps
    load_best_model_at_end=True,               # Load the best model after training
    fp16=False,                                # Do not use mixed precision on CPU
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./models/fine_tuned_gpt_nl")
tokenizer.save_pretrained("./models/fine_tuned_gpt_nl")
