import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from src.dataset import MethylationDataset
from src.train import FibrosisClassifier

# Load test data
test_dataset = MethylationDataset('data/processed/X_test.npy', 'data/processed/y_test.npy')
test_loader = DataLoader(test_dataset, batch_size=32)

# Load model from checkpoint — Lightning saves these automatically
model = FibrosisClassifier.load_from_checkpoint(r'C:\compbio\OA-scRNA-severity-nn\lightning_logs\version_3\checkpoints\epoch=9-step=400.ckpt')
model.eval()

# Get predictions
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        logits = model(x)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(y.numpy())

# Per class F1
print(classification_report(all_labels, all_preds, target_names=['S0','S1','S2','S3','S4']))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['S0','S1','S2','S3','S4'], yticklabels=['S0','S1','S2','S3','S4'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('results/figures/confusion_matrix.png')
plt.show()