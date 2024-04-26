import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

def main():
    # Load and preprocess the dataset
    df = pd.read_csv('train_DNABERT.csv')
    df['Embedding'] = df['Embedding'].apply(lambda x: np.fromstring(x.strip("[]"), sep=',', dtype=np.float32))
    label_encoder = LabelEncoder()
    df['Feature'] = label_encoder.fit_transform(df['Feature'])
    X = np.stack(df['Embedding'].values)
    y = df['Feature'].values
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Split and create DataLoader instances
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Set up device and model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    num_classes = len(label_encoder.classes_)
    model = NN(input_size=768, num_classes=num_classes).to(device)

    # Loss, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    train_losses = []
    val_losses = []

    for epoch in range(75):
        model.train()
        train_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                tepoch.set_postfix(loss=loss.item())

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)

        # Update the learning rate based on the validation loss
        scheduler.step(val_loss)

    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder
    }, 'model.pth')

    # Print label encoder mapping
    print("Label Encoder Mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"{i}: {label}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print(f'Accuracy: {100 * correct / total}%')

    # Report confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Classification report
    target_names = label_encoder.classes_
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.show()

if __name__ == "__main__":
    main()