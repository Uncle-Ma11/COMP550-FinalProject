import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load GoEmotions dataset
dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
num_classes = 28

# Preprocess and Map Labels for Multi-Label Classification
def preprocess_and_map_labels(example):
    label_array = np.zeros(num_classes, dtype=np.float32)
    for label in example["labels"]:
        label_array[label] = 1.0
    example["label_array"] = label_array
    return example

dataset = dataset.map(preprocess_and_map_labels)

train_texts = [d["text"] for d in dataset["train"]]
train_labels = [d["label_array"] for d in dataset["train"]]
test_texts = [d["text"] for d in dataset["test"]]
test_labels = [d["label_array"] for d in dataset["test"]]

# Vectorize Text
vectorizer = CountVectorizer(max_features=5000)
train_encodings = vectorizer.fit_transform(train_texts).toarray()
test_encodings = vectorizer.transform(test_texts).toarray()

# Custom Dataset Class
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.encodings[idx], self.labels[idx]

train_data = SentimentDataset(train_encodings, train_labels)
test_data = SentimentDataset(test_encodings, test_labels)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# CNN Model
class CNNTextClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_filters, filter_sizes):
        super(CNNTextClassifier, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = [F.relu(conv(x)).squeeze(2) for conv in self.convs]
        x = [F.max_pool1d(feature_map, feature_map.size(2)).squeeze(2) for feature_map in x]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

# Initialize Model
input_dim = train_encodings.shape[1]
num_filters = 100
filter_sizes = [3, 4, 5]

model = CNNTextClassifier(input_dim, num_classes, num_filters, filter_sizes).to(device)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()  # Multi-label binary cross-entropy
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Training Loop with Loss and Accuracy Tracking
def train_model(model, train_loader, optimizer, criterion, device, epochs=5):
    model.train()
    train_losses = []
    train_accuracies = []
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for encodings, labels in train_loader:
            # Convert encodings and labels to correct tensor formats
            encodings = torch.tensor(encodings, dtype=torch.float32).to(device)
            labels = torch.tensor(np.array(labels), dtype=torch.float32).to(device)

            # Ensure labels are shaped as (batch_size, num_classes)
            labels = labels.view(encodings.size(0), -1)

            optimizer.zero_grad()
            outputs = model(encodings)

            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()

        avg_loss = total_loss / len(train_loader)
        avg_accuracy = correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    return train_losses, train_accuracies


# Evaluation Loop
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for encodings, labels in test_loader:
            encodings = torch.tensor(encodings, dtype=torch.float32).to(device)
            labels = torch.tensor(np.array(labels), dtype=torch.float32).to(device)

            outputs = model(encodings)
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5  # Apply threshold
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


# Plotting Functions
def plot_training_loss_accuracy(losses, accuracies):
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label="Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_multilabel(labels, preds, num_classes):
    for i in range(num_classes):
        cm = confusion_matrix(labels[:, i], preds[:, i])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix for Class {i}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()


def plot_true_vs_predicted(labels, preds):
    true_counts = Counter(labels.argmax(axis=1))
    pred_counts = Counter(preds.argmax(axis=1))

    plt.figure(figsize=(12, 6))
    plt.bar(true_counts.keys(), true_counts.values(), alpha=0.5, label="True")
    plt.bar(pred_counts.keys(), pred_counts.values(), alpha=0.5, label="Predicted")
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.title("True vs Predicted Label Distribution")
    plt.legend()
    plt.show()

def plot_roc_curve_multilabel(labels, preds, num_classes):
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels[:, i], preds[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Per Class)")
    plt.legend(loc="lower right")
    plt.show()


# Train and Evaluate
train_losses, train_accuracies = train_model(model, train_loader, optimizer, criterion, device, epochs=5)
labels, preds = evaluate_model(model, test_loader, device)

# Visualize Results
plot_training_loss_accuracy(train_losses, train_accuracies)
plot_confusion_matrix_multilabel(labels, preds)
plot_true_vs_predicted(labels, preds)
plot_roc_curve_multilabel(labels, preds, num_classes)
