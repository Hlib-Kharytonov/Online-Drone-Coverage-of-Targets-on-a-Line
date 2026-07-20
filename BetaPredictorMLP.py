import torch.nn as nn

class BetaPredictorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128) 
        self.fc2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.output(x)