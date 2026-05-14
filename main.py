# main.py
import os
import sys

from src.train import FibrosisClassifier
from src.dataset import scRNADataset
from torch.utils.data import DataLoader
import lightning as L

if __name__ == "__main__":
    train_dataset = scRNADataset('data/processed/X_train.npy', 'data/processed/y_train.npy')
    test_dataset = scRNADataset('data/processed/X_test.npy', 'data/processed/y_test.npy')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)

    classifier = FibrosisClassifier()

    trainer = L.Trainer(max_epochs=10, log_every_n_steps=10)
    trainer.fit(classifier, train_loader, val_loader)