import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load GoEmotions dataset
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

# Define the CNN Model
def create_model(num_classes, dropout_rate):
    class CNNTextClassifier(nn.Module):
        def __init__(self):
            super(CNNTextClassifier, self).__init__()
            self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(128 * 5000, 256)  # Adjust size according to actual features
            self.fc2 = nn.Linear(256, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout_rate)
            self.batchnorm1 = nn.BatchNorm1d(64)
            self.batchnorm2 = nn.BatchNorm1d(128)

        def forward(self, x):
            x = x.unsqueeze(1)  # Add channel dimension
            x = self.relu(self.batchnorm1(self.conv1(x)))
            x = self.relu(self.batchnorm2(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.fc2(x)
            return x
    return CNNTextClassifier().to(device)

# Experiment Configurations
experiment_params = [
    {"param": "lr", "values": [0.01, 0.001, 0.0001]},
    {"param": "batch_size", "values": [16, 32, 64]},
    {"param": "dropout_rate", "values": [0.3, 0.5, 0.7]},
    {"param": "optimizer", "values": ["adam", "sgd"]}
]

# Run and evaluate models
def run_experiments():
    all_results = {}
    base_config = {"lr": 0.001, "batch_size": 32, "dropout_rate": 0.5, "optimizer": "adam"}
    for experiment in experiment_params:
        param = experiment['param']
        results = []
        for value in experiment['values']:
            config = base_config.copy()
            config[param] = value
            model = create_model(num_classes, config['dropout_rate'])
            train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
            test_loader = DataLoader(test_data, batch_size=config['batch_size'])
            optimizer = optim.Adam(model.parameters(), lr=config['lr']) if config['optimizer'] == 'adam' else optim.SGD(model.parameters(), lr=config['lr'])
            criterion = nn.BCEWithLogitsLoss()
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

            # Training loop
            epoch_losses, epoch_accuracies = [], []
            for epoch in range(10):
                total_loss, total_correct, total_examples = 0, 0, 0
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    total_correct += (predicted == labels).float().sum()
                    total_examples += labels.numel()

                accuracy = total_correct / total_examples
                epoch_losses.append(total_loss / len(train_loader))
                epoch_accuracies.append(accuracy.item())

                print(f"Config {config}: Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}, Accuracy: {accuracy.item()}")

                scheduler.step()

            # Evaluation
            model.eval()
            total_preds, total_labels = [], []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    predictions = torch.sigmoid(outputs)
                    total_preds.append(predictions.cpu().numpy())
                    total_labels.append(labels.cpu().numpy())
            all_preds = np.concatenate(total_preds, axis=0)
            all_labels = np.concatenate(total_labels, axis=0)
            report = classification_report(all_labels, all_preds > 0.5, zero_division=1)
            test_accuracy = accuracy_score(all_labels, all_preds > 0.5)

            results.append({
                "config": config,
                "losses": epoch_losses,
                "accuracies": epoch_accuracies,
                "classification_report": report,
                "test_accuracy": test_accuracy
            })
            print(f"Completed: {config}")

        all_results[param] = results

    return all_results

# Visualize results
def visualize_results(all_results):
    # Individual parameter visualizations
    for param, results in all_results.items():
        plt.figure(figsize=(14, 7))
        for result in results:
            plt.plot(result['accuracies'], label=f"{param} {result['config'][param]} (Test Acc: {result['test_accuracy']:.2f})")
        plt.title(f'Training and Testing Accuracy for varying {param}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    # Consolidated visualization
    plt.figure(figsize=(14, 7))
    for param, results in all_results.items():
        for result in results:
            plt.plot(result['accuracies'], label=f"{param} {result['config'][param]} (Test Acc: {result['test_accuracy']:.2f})")
    plt.title('Consolidated Training and Testing Accuracy Across All Configurations')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


results = run_experiments()
visualize_results(results)
