import torch.nn as nn

class BetaPredictorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Распаковка (4 -> 64)
        self.fc1 = nn.Linear(4, 64) 
        
        # Расширение и глубокий анализ (64 -> 128)
        self.fc2 = nn.Linear(64, 128)
        
        # Сжатие и фильтрация шума (128 -> 64)
        self.fc3 = nn.Linear(128, 64)
        
        # Финальный вывод (64 -> 1)
        self.output = nn.Linear(64, 1)
        
        self.activation = nn.ReLU() 
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return self.output(x)