import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from datasets import load_dataset

import evaluate

# === CONFIGURATION ===
MODEL_NAME = "chandar-lab/NeoBERT"
NUM_LABELS = 6  # For LIAR2 dataset
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5

# === PATHS ===
BASE_DIR = "C:/Users/Zayed/Documents/GitHub/FactuAI/data/processed"
train_path = os.path.join(BASE_DIR, "liar2_train.csv")
val_path = os.path.join(BASE_DIR, "liar2_val.csv")
test_path = os.path.join(BASE_DIR, "liar2_test.csv")

# === LOAD TOKENIZER AND MODEL ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, trust_remote_code=True)

# === LOAD AND TOKENIZE DATASET ===
dataset = load_dataset("csv", data_files={
    "train": train_path,
    "validation": val_path,
    "test": test_path
})

def tokenize_function(examples):
    return tokenizer(examples["statement"], padding="max_length", truncation=True, max_length=MAX_LENGTH)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# === DEFINE METRICS ===
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# === TRAINING SETUP ===
training_args = TrainingArguments(
    output_dir="./neo-bert-finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    fp16=True,  # Enable mixed precision training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# === TRAIN ===
trainer.train()

# === FINAL EVALUATION ===
results = trainer.evaluate(tokenized_datasets["test"])
print("Test Results:", results)
