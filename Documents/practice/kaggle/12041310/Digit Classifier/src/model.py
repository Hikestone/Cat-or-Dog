import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import datasets, transforms
from PIL import Image

class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding="same") 
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

class MNISTDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = torch.tensor(self.data.iloc[idx, 0], dtype=torch.long)
        image = torch.tensor(self.data.iloc[idx, 1:].values, dtype=torch.float32).view(1, 28, 28) / 255.0
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, label


class Trainer:
    def __init__(self, root_path: str, patience: int = 3):
        # Define transformations for data augmentation and normalization
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )
        self.root = root_path
        # Initialize the model, loss function, and optimizer
        self.model = DigitClassifier()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.patience = patience
        self.best_val_loss = float('inf')
        self.current_patience = 0

    def _load_data(self):
        mnist_df_train = pd.read_csv('../input/train.csv')
        mnist_df_test = pd.read_csv('../input/test.csv')
        self.train_images = MNISTDataset(mnist_df_train, transform=self.transform)
        self.val_images = MNISTDataset(mnist_df_test, transform=self.transform)

    def _extract_data(self):
        self.train_loader = DataLoader(self.train_images, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_images, batch_size=32, shuffle=False)

    def _eval(self):
        self.model.eval()
        total_val_loss = 0
        correct_val_predictions = 0
        total_val_samples = 0

        with torch.no_grad():
            for val_inputs, val_labels in self.val_loader:
                val_outputs = self.model(val_inputs)
                val_loss = self.criterion(
                    val_outputs, val_labels.type(torch.LongTensor)
                )

                total_val_loss += val_loss.item()
                val_predicted = val_outputs.argmax(dim=1, keepdim=True)
                correct_val_predictions += val_predicted.eq(val_labels.view_as(val_predicted)).sum().item()
                total_val_samples += val_labels.size(0)

        val_loss = total_val_loss / len(self.val_loader)
        val_accuracy = correct_val_predictions / total_val_samples
        return val_loss, val_accuracy

    def _train(self):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()
            # print(labels)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.type(torch.LongTensor))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predicted = outputs.argmax(dim=1, keepdim=True)
            # print(predicted)
            correct_predictions += predicted.eq(labels.view_as(predicted)).sum().item()
            # print(correct_predictions)
            total_samples += labels.size(0)
            # break

        epoch_loss = total_loss / len(self.train_loader)
        epoch_accuracy = correct_predictions / total_samples
        return epoch_loss, epoch_accuracy

    def train(self, num_epochs: int = 20):
        self._load_data()
        self._extract_data()
        
        for epoch in range(num_epochs):
            train_loss, train_accuracy = self._train()
            print(
                f"Training - Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}"
            )
            val_loss, val_accuracy = self._eval()
            print(
                f"Validation - Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
            )
            # break
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.current_patience = 0
            else:
                self.current_patience += 1

            if self.current_patience >= self.patience:
                print(f"Early stopping at epoch {epoch + 1} due to lack of improvement in validation loss.")
                break


    def save(self, save_path: str):
        torch.save(self.model.state_dict(), save_path)


if __name__ == "__main__":
    trainer = Trainer(root_path="input")
    trainer.train()
    trainer.save(save_path="../model/digit_classifier.pth")
