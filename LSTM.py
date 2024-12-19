import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer
from tqdm import tqdm
import string

# Initialize tokenizer and preprocessing tools
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Preprocessing function
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Split words, lemmatize, and remove stopwords
    words = text.split()
    cleaned_text = " ".join([lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words])
    return cleaned_text


# Custom PyTorch Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=max_len,
                                   return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }

    def __len__(self):
        return len(self.labels)


# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        packed_output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # Concatenate forward and backward LSTM outputs
        output = self.fc(self.dropout(hidden))
        return output


# Metrics Function
def compute_metrics(preds, labels):
    preds = torch.argmax(preds, axis=1).cpu().numpy()
    labels = labels.cpu().numpy()
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    DATASET = int(input("Enter 1 for DAIR AI or 2 for Go Emotions: "))

    # Load and preprocess datasets
    if DATASET == 1:
        # DAIR AI Dataset
        dataset = load_dataset("dair-ai/emotion", "split")


        def preprocess_labels(example):
            example["text"] = preprocess_text(example["text"])
            example["label"] = example["label"]  # Already integer labels
            return example


        dataset = dataset.map(preprocess_labels)
        output_dim = 6  # 6 classes
        train_texts = dataset["train"]["text"]
        train_labels = dataset["train"]["label"]
        test_texts = dataset["test"]["text"]
        test_labels = dataset["test"]["label"]

    elif DATASET == 2:
        # Go Emotions Dataset
        dataset = load_dataset("google-research-datasets/go_emotions", "simplified")


        def preprocess_and_map_labels(example):
            example["text"] = preprocess_text(example["text"])
            example["label"] = example["labels"][0] if len(example["labels"]) > 0 else -1  # Use the first label
            return example


        dataset = dataset.map(preprocess_and_map_labels)
        dataset = dataset.filter(lambda x: x["label"] != -1)
        output_dim = 28
        train_texts = dataset["train"]["text"]
        train_labels = [label for label in dataset["train"]["label"]]
        test_texts = dataset["test"]["text"]
        test_labels = [label for label in dataset["test"]["label"]]

    else:
        raise ValueError("Invalid choice. Enter 1 for DAIR AI or 2 for Go Emotions.")

    print("\nSample Train Set Examples:")
    for i in range(3):  # Print 3 examples
        print(f"Text: {train_texts[i]}")
        print(f"Label: {train_labels[i]}\n")

    print("Sample Test Set Examples:")
    for i in range(3):  # Print 3 examples
        print(f"Text: {test_texts[i]}")
        print(f"Label: {test_labels[i]}\n")

    # Create Datasets and DataLoaders
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Initialize LSTM Model
    vocab_size = tokenizer.vocab_size
    embedding_dim = 128
    hidden_dim = 256
    pad_idx = tokenizer.pad_token_id

    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training Loop with Progress Bar
    for epoch in range(5):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=True)
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

    # Evaluation Loop with Progress Bar
    model.eval()
    all_preds, all_labels = [], []
    loop = tqdm(test_loader, desc="Evaluating", leave=True)
    with torch.no_grad():
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            all_preds.append(outputs)
            all_labels.append(labels)

    # Compute Metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_preds, all_labels)
    print("Evaluation Results:", metrics)
