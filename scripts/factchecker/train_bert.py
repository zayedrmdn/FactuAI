import os
import json
import logging
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import torch
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from datasets import load_dataset

# === SETUP LOGGING ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 6
MAX_LENGTH = 256
USE_FP16 = True
SEED = 42

set_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === BEST HYPERPARAMETERS ===
LEARNING_RATE = 3.146714266763938e-05
BATCH_SIZE = 16
WEIGHT_DECAY = 0.13246527431263422
EPOCHS = 10
RUN_ID = f"bert_final_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# === PATHS ===
BASE_DIR = "C:/Users/Zayed/Documents/GitHub/FactuAI/data/processed"
train_path = os.path.join(BASE_DIR, "liar2_train.csv")
val_path   = os.path.join(BASE_DIR, "liar2_val.csv")
test_path  = os.path.join(BASE_DIR, "liar2_test.csv")

OUTPUT_DIR = f"D:/Checkpoints/bert/{RUN_ID}"
LOG_DIR    = f"D:/Checkpoints/bert/{RUN_ID}/logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === VERIFY DATA FILES ===
for path, name in [(train_path, "train"), (val_path, "validation"), (test_path, "test")]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name.capitalize()} file not found: {path}")
    logger.info(f"Found {name} file: {path}")

# === LOAD TOKENIZER ===
logger.info(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# === LOAD DATASETS ===
logger.info("Loading datasets...")
dataset = load_dataset("csv", data_files={
    "train": train_path,
    "validation": val_path,
    "test": test_path
})

for split in dataset.keys():
    logger.info(f"{split.capitalize()} set size: {len(dataset[split])}")

# === ADD COMPOSITE TEXT FIELD ===
def add_combined_text(example):
    return {
        "text": (
            f"[SPEAKER] {example['speaker']} "
            f"[DESC] {example['speaker_description']} "
            f"[SUBJECT] {example['subject']} "
            f"[CONTEXT] {example['context']} "
            f"[STATEMENT] {example['statement']} "
            f"[JUSTIFICATION] {example['justification']}"
        )
    }

logger.info("Adding combined text field...")
dataset = dataset.map(add_combined_text)

# === TOKENIZATION FUNCTION ===
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

logger.info("Tokenizing datasets...")
tokenized_datasets = dataset.map(tokenize_function, batched=True, desc="Tokenizing")

# === METRICS SETUP ===
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1_macro": f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"],
        "f1_weighted": f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    }

# === LOAD MODEL ===
logger.info(f"Loading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

# === TRAINING ARGUMENTS ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=500,
    logging_dir=LOG_DIR,
    logging_steps=50,
    seed=SEED,
    fp16=USE_FP16,
    report_to="tensorboard",
    push_to_hub=False
)

# === SAVE CONFIGURATION ===
config = {
    "model_name": MODEL_NAME,
    "num_labels": NUM_LABELS,
    "max_length": MAX_LENGTH,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "weight_decay": WEIGHT_DECAY,
    "epochs": EPOCHS,
    "seed": SEED,
    "fp16": USE_FP16,
    "run_id": RUN_ID,
    "timestamp": datetime.now().isoformat()
}
with open(os.path.join(OUTPUT_DIR, "training_config.json"), 'w') as f:
    json.dump(config, f, indent=2)

# === INITIALIZE TRAINER ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# === TRAINING ===
logger.info("Starting training...")
train_result = trainer.train()
with open(os.path.join(OUTPUT_DIR, "train_results.json"), 'w') as f:
    json.dump(train_result.metrics, f, indent=2)

# === EVALUATION FUNCTION ===
def detailed_evaluation(name, dataset):
    results = trainer.evaluate(dataset)
    predictions = trainer.predict(dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids
    report = classification_report(y_true, y_pred, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)
    eval_results = {
        "basic_metrics": results,
        "classification_report": report,
        "confusion_matrix": matrix.tolist()
    }
    with open(os.path.join(OUTPUT_DIR, f"{name}_detailed_results.json"), 'w') as f:
        json.dump(eval_results, f, indent=2)
    return eval_results

# === RUN EVALUATION ===
val_results = detailed_evaluation("validation", tokenized_datasets["validation"])
test_results = detailed_evaluation("test", tokenized_datasets["test"])

# === SAVE FINAL MODEL ===
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)

# === SUMMARY ===
summary = {
    "model_name": MODEL_NAME,
    "run_id": RUN_ID,
    "training_completed": True,
    "validation_accuracy": val_results["basic_metrics"]["validation_accuracy"],
    "validation_f1_weighted": val_results["basic_metrics"]["validation_f1_weighted"],
    "test_accuracy": test_results["basic_metrics"]["test_accuracy"],
    "test_f1_weighted": test_results["basic_metrics"]["test_f1_weighted"],
    "model_path": OUTPUT_DIR,
    "timestamp": datetime.now().isoformat()
}
with open(os.path.join(OUTPUT_DIR, "training_summary.json"), 'w') as f:
    json.dump(summary, f, indent=2)

print("\n🎉 Training complete")
print(f"📁 Model saved to: {OUTPUT_DIR}")
print(f"📊 Test F1-Weighted: {summary['test_f1_weighted']:.4f}")
