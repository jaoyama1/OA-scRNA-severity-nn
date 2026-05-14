import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], num_classes)
        )
    
    def forward(self, x):
        return self.network(x)