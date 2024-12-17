from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

DATASET = 2

go_emotions_ds = load_dataset("google-research-datasets/go_emotions", "simplified")

dair_ai_ds = load_dataset("dair-ai/emotion", "split")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    words = text.split()
    cleaned_text = " ".join([lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words])
    return cleaned_text


def tokenize_data(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(pred):
    preds = torch.argmax(torch.tensor(pred.predictions), axis=1)
    labels = torch.tensor(pred.label_ids)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    if DATASET == 1:
        dair_ai_ds = dair_ai_ds.map(lambda x: {"text": preprocess_text(x["text"])})
        dataset = dair_ai_ds
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)
    elif DATASET == 2:
        def preprocess_and_map_labels(example):
            example["text"] = preprocess_text(example["text"])
            example["label"] = example["labels"][0] if len(example["labels"]) > 0 else -1  # Use the first label or -1
            return example

        # Apply preprocessing
        go_emotions_ds = go_emotions_ds.map(preprocess_and_map_labels)

        # Filter invalid labels
        go_emotions_ds = go_emotions_ds.filter(lambda x: x["label"] != -1)

        dataset = go_emotions_ds
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28)  # 28 emotions
    else:
        raise ValueError("Invalid DATASET value. Choose 1 for DAIR AI or 2 for Go Emotions.")

    # Tokenize the dataset
    dataset = dataset.map(tokenize_data, batched=True)

    # Set format for PyTorch
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Split dataset into train and test
    train_data = dataset["train"]
    test_data = dataset["test"]

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save the model at the end of each epoch
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,  # Keep only the 2 most recent checkpoints
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True  # Load the best model after training
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    results = trainer.evaluate()
    print("Evaluation results:", results)
