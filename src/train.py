import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.dataset import scRNADataset
from src.model import MLP


class FibrosisClassifier(L.LightningModule):
    def __init__(self, input_dim=5000, hidden_dims=[128, 32], num_classes=5, lr=1e-3):
        super().__init__()
        self.model = MLP(input_dim, hidden_dims, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    # Datasets
    train_dataset = scRNADataset('data/processed/X_train.npy', 'data/processed/y_train.npy')
    test_dataset = scRNADataset('data/processed/X_test.npy', 'data/processed/y_test.npy')

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32)

    # Model
    classifier = FibrosisClassifier()

    # Trainer
    trainer = L.Trainer(max_epochs=20, log_every_n_steps=10)
    trainer.fit(classifier, train_loader, val_loader)