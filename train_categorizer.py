"""Fine-tune DistilBERT for receipt item categorization."""

import json
import os
import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

CATEGORIES = [
    "groceries", "dining", "transport", "entertainment",
    "health", "clothing", "utilities", "other",
]
LABEL2ID = {cat: i for i, cat in enumerate(CATEGORIES)}
ID2LABEL = {i: cat for i, cat in enumerate(CATEGORIES)}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "categorizer")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")


def load_data():
    """Load the synthetic dataset."""
    if not os.path.exists(DATASET_PATH):
        print("Dataset not found. Generating...")
        from generate_dataset import generate_dataset
        generate_dataset()

    with open(DATASET_PATH) as f:
        data = json.load(f)

    # Convert to HuggingFace dataset
    texts = [d["text"] for d in data]
    labels = [LABEL2ID[d["label"]] for d in data]

    dataset = Dataset.from_dict({"text": texts, "label": labels})

    # Train/test split
    split = dataset.train_test_split(test_size=0.2, seed=42)
    return split["train"], split["test"]


def train():
    """Fine-tune DistilBERT on the receipt item dataset."""
    print("Loading tokenizer and model...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(CATEGORIES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    print("Loading dataset...")
    train_dataset, eval_dataset = load_data()

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=64,
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting fine-tuning...")
    trainer.train()

    # Save model and tokenizer
    print(f"Saving model to {MODEL_DIR}")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # Evaluate
    results = trainer.evaluate()
    print(f"\nEvaluation results:")
    print(f"  Loss: {results['eval_loss']:.4f}")
    print(f"  Accuracy: {results['eval_accuracy']:.4f}")

    return results


if __name__ == "__main__":
    train()
