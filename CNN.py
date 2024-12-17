import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load dataset
dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
num_classes = 28

# Preprocess and map labels
def preprocess_and_map_labels(example):
    label_array = np.zeros(num_classes, dtype=np.float32)
    for label in example["labels"]:
        label_array[label] = 1.0
    example["label_array"] = label_array
    return example

dataset = dataset.map(preprocess_and_map_labels)

# Extract train and test data
train_texts = [d["text"] for d in dataset["train"]]
train_labels = [d["label_array"] for d in dataset["train"]]
test_texts = [d["text"] for d in dataset["test"]]
test_labels = [d["label_array"] for d in dataset["test"]]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
train_encodings = vectorizer.fit_transform(train_texts).toarray()
test_encodings = vectorizer.transform(test_texts).toarray()

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.encodings[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

train_data = SentimentDataset(train_encodings, train_labels)
test_data = SentimentDataset(test_encodings, test_labels)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# CNN Model with improved architecture
class CNNTextClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CNNTextClassifier, self).__init__()
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=7, padding=3)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(256 * num_features // 2, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = CNNTextClassifier(num_features=5000, num_classes=num_classes).to(device)

# Class weights for imbalanced dataset
class_weights = torch.tensor(np.max(np.sum(train_labels, axis=0)) / np.sum(train_labels, axis=0), dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Train the model
def train_model(model, train_loader, criterion, optimizer, scheduler, device, epochs=60):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')

train_model(model, train_loader, criterion, optimizer, scheduler, device, epochs=60)

# Evaluate the model
def evaluate_model(model, test_loader, device):
    model.eval()
    total_preds = []
    total_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = torch.sigmoid(outputs)
            total_preds.append(predictions.cpu().numpy())
            total_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(total_preds, axis=0)
    all_labels = np.concatenate(total_labels, axis=0)
    thresholds = [0.5] * num_classes  # Placeholder for dynamic threshold calculation
    preds = (all_preds > thresholds).astype(int)
    print(classification_report(all_labels, preds, zero_division=1))  # Handle undefined metrics by assuming perfect classification for missing labels

evaluate_model(model, test_loader, device)