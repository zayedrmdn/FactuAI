# train_distilbert.py

import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import evaluate

# =========================
# === CONFIGURATION ===
# =========================
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 6  # For LIAR2 dataset
MAX_LENGTH = 128
BATCH_SIZE = 32             # faster training 
EPOCHS = 5                  # 
LEARNING_RATE = 3e-5        # equals to 0.00003, a common choice for fine-tuning
USE_FP16 = True 

# =========================
# === FILE PATHS ===
# =========================
BASE_DIR = "C:/Users/Zayed/Documents/GitHub/FactuAI/data/processed"
train_path = os.path.join(BASE_DIR, "liar2_train.csv")
val_path = os.path.join(BASE_DIR, "liar2_val.csv")
test_path = os.path.join(BASE_DIR, "liar2_test.csv")

# =========================
# === LOAD TOKENIZER & MODEL ===
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# =========================
# === LOAD AND TOKENIZE DATASET ===
# =========================
dataset = load_dataset("csv", data_files={
    "train": train_path,
    "validation": val_path,
    "test": test_path
})

def tokenize_function(example):
    return tokenizer(example["statement"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# =========================
# === METRIC FUNCTION ===
# =========================
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# =========================
# === TRAINING ARGUMENTS ===
# =========================
training_args = TrainingArguments(
    output_dir="C:/Users/Zayed/Documents/GitHub/FactuAI/models",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    fp16=USE_FP16
)

# =========================
# === TRAINER SETUP ===
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# =========================
# === TRAIN MODEL ===
# =========================
trainer.train()

# =========================
# === FINAL EVALUATION ===
# =========================
results = trainer.evaluate(tokenized_datasets["test"])
print("Test Results:", results)
