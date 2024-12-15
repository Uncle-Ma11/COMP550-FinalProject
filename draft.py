ok let take one step back, before for this model my result:

Using device: cuda
Epoch 1, Loss: 0.2141, Accuracy: 0.8916
Epoch 2, Loss: 0.0180, Accuracy: 0.9546
Epoch 3, Loss: 0.0113, Accuracy: 0.9570
Epoch 4, Loss: 0.0100, Accuracy: 0.9574
Epoch 5, Loss: 0.0092, Accuracy: 0.9577
Epoch 6, Loss: 0.0087, Accuracy: 0.9578
Epoch 7, Loss: 0.0084, Accuracy: 0.9578
Epoch 8, Loss: 0.0082, Accuracy: 0.9579
Epoch 9, Loss: 0.0081, Accuracy: 0.9579
Epoch 10, Loss: 0.0081, Accuracy: 0.9579
Epoch 11, Loss: 0.0080, Accuracy: 0.9579
Epoch 12, Loss: 0.0080, Accuracy: 0.9579
Epoch 13, Loss: 0.0079, Accuracy: 0.9579
Epoch 14, Loss: 0.0079, Accuracy: 0.9579
Epoch 15, Loss: 0.0079, Accuracy: 0.9579
Epoch 16, Loss: 0.0079, Accuracy: 0.9579
Epoch 17, Loss: 0.0079, Accuracy: 0.9579
Epoch 18, Loss: 0.0078, Accuracy: 0.9579
Epoch 19, Loss: 0.0078, Accuracy: 0.9579
Epoch 20, Loss: 0.0079, Accuracy: 0.9579
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       504
           1       0.06      0.00      0.01       264
           2       0.00      0.00      0.00       198
           3       0.00      0.00      0.00       320
           4       0.00      0.00      0.00       351
           5       0.00      0.00      0.00       135
           6       0.00      0.00      0.00       153
           7       0.06      0.00      0.01       284
           8       0.00      0.00      0.00        83
           9       0.00      0.00      0.00       151
          10       0.00      0.00      0.00       267
          11       0.00      0.00      0.00       123
          12       0.00      0.00      0.00        37
          13       0.00      0.00      0.00       103
          14       0.00      0.00      0.00        78
          15       0.00      0.00      0.00       352
          16       0.00      0.00      0.00         6
          17       0.00      0.00      0.00       161
          18       0.00      0.00      0.00       238
          19       0.00      0.00      0.00        23
          20       0.00      0.00      0.00       186
          21       0.00      0.00      0.00        16
          22       0.00      0.00      0.00       145
          23       0.00      0.00      0.00        11
          24       0.00      0.00      0.00        56
          25       0.00      0.00      0.00       156
          26       0.00      0.00      0.00       141
          27       0.94      0.01      0.02      1787

   micro avg       0.04      0.00      0.01      6329
   macro avg       0.04      0.00      0.00      6329
weighted avg       0.27      0.00      0.01      6329
 samples avg       1.00      0.00      0.00      6329
how do i improve



fix this model don't change the name i want to not impact my cdode after that can still run the same but only change the CNN archetecture for sentiment analysis NLP task:
# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

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

# Extract train and test splits
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
        return torch.tensor(self.encodings[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Prepare Datasets
train_data = SentimentDataset(train_encodings, train_labels)
test_data = SentimentDataset(test_encodings, test_labels)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# CNN Model
class CNNTextClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_filters, filter_sizes):
        super(CNNTextClassifier, self).__init__()
        
        # Convolutional layers for different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=fs, padding=fs // 2)
            for fs in filter_sizes
        ])
        
        # Batch Normalization layers to stabilize training
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in filter_sizes
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 256)  # First fully connected layer
        self.fc2 = nn.Linear(256, 128)  # Second fully connected layer
        self.fc3 = nn.Linear(128, num_classes)  # Final output layer

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Add channel dimension for CNN (batch_size, 1, sequence_length)
        x = x.unsqueeze(1)
        
        # Convolutional layers with batch normalization, activation, and global max pooling
        conv_results = []
        for conv, bn in zip(self.convs, self.batch_norms):
            conv_out = self.relu(bn(conv(x)))
            pooled_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # Global max pooling
            conv_results.append(pooled_out)
        
        # Concatenate features from all filters
        x = torch.cat(conv_results, dim=1)  # Shape: (batch_size, len(filter_sizes) * num_filters)
        
        # Fully connected layers with dropout and activation
        x = self.dropout(self.relu(self.fc1(x)))  # Layer 1
        x = self.dropout(self.relu(self.fc2(x)))  # Layer 2
        logits = self.fc3(x)  # Final output layer
        
        return logits

# Initialize Model
input_dim = train_encodings.shape[1]
num_filters = 100
filter_sizes = [3, 4, 5]

model = CNNTextClassifier(input_dim, num_classes, num_filters, filter_sizes).to(device)

# Normalize class weights for imbalanced dataset
class_weights = 1.0 / (np.sum(train_labels, axis=0) + 1e-6)  # Avoid division by zero
class_weights = class_weights / np.sum(class_weights)  # Normalize to sum to 1
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

# Training Function
def train_model(model, train_loader, optimizer, criterion, device, scheduler=None, epochs=5):
    model.train()
    train_losses = []
    train_accuracies = []
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for encodings, labels in train_loader:
            encodings = encodings.to(device)
            labels = labels.to(device)
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

        if scheduler:
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        avg_accuracy = correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    return train_losses, train_accuracies

# Compute dynamic thresholds
def compute_dynamic_thresholds(train_loader, model, device):
    model.eval()
    all_train_preds = []
    with torch.no_grad():
        for encodings, _ in train_loader:
            encodings = encodings.to(device)
            outputs = torch.sigmoid(model(encodings))
            all_train_preds.append(outputs.cpu().numpy())

    all_train_preds = np.vstack(all_train_preds)
    return np.mean(all_train_preds, axis=0)

# Evaluation Function with Dynamic Thresholds
def evaluate_model_with_dynamic_thresholds(model, test_loader, device, dynamic_thresholds):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for encodings, labels in test_loader:
            encodings = encodings.to(device)
            labels = labels.to(device)

            outputs = model(encodings)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Apply dynamic thresholds
    binarized_preds = (all_preds > dynamic_thresholds).astype(int)

    # Generate classification report
    report = classification_report(all_labels, binarized_preds, zero_division=1, output_dict=True)
    print(classification_report(all_labels, binarized_preds, zero_division=1))

    return all_labels, all_preds, report

# Visualization: Per-Class Metrics Bar Plot
def plot_per_class_metrics(report, num_classes):
    precision = [report[str(i)]['precision'] for i in range(num_classes)]
    recall = [report[str(i)]['recall'] for i in range(num_classes)]
    f1 = [report[str(i)]['f1-score'] for i in range(num_classes)]

    # Bar plot
    x = np.arange(num_classes)
    width = 0.25

    plt.figure(figsize=(15, 7))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-Score')

    plt.xlabel("Classes")
    plt.ylabel("Scores")
    plt.title("Per-Class Metrics")
    plt.legend()
    plt.show()

# Train the Model
train_losses, train_accuracies = train_model(model, train_loader, optimizer, criterion, device, scheduler, epochs=20)

# Compute thresholds
dynamic_thresholds = compute_dynamic_thresholds(train_loader, model, device)

# Evaluate the Model
labels, preds, report = evaluate_model_with_dynamic_thresholds(model, test_loader, device, dynamic_thresholds)

# Visualizations
plot_per_class_metrics(report, num_classes)
