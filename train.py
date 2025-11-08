import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam, SGD

class Trainer:
    """
    Trainer class to handle training and evaluation of the model.
    """
    def __init__(self, model, train_loader, val_loader, optimizer, epochs=10):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.MSELoss(reduce='sum') 
        self.optimizer = optimizer
        self.epochs = epochs

    def train(self):
        for epoch_num in range(1, self.epochs + 1):
            train_loss = self.training_step()
            val_loss = self.evaluation_step()
            print(f"Epoch {epoch_num}/{self.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        

    def training_step(self):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc="Training step")

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs.squeeze(), targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            pbar.update(1)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def evaluation_step(self):
        self.model.eval()
        running_loss = 0.0
        pbar = tqdm(self.val_loader, desc="Testing step")

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), targets)
                running_loss += loss.item() * inputs.size(0)
                pbar.update(1)
        epoch_loss = running_loss / len(self.val_loader.dataset)
        return epoch_loss