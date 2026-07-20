import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class DroneDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
        # Масштабируем базовые признаки (как мы делали раньше)
        n_scaled = self.data['n'].values / 1000.0
        mu_scaled = self.data['mu'].values / 1000.0
        sigma_scaled = self.data['sigma'].values / 1000.0
        
        # Вычисляем новый признак CV = sigma / mu
        cv_raw = self.data['sigma'].values / (self.data['mu'].values + 1.0)
        cv_scaled = np.clip(cv_raw, 0, 20.0) / 20.0 
        
        # Обновляем сборку матрицы:
        self.X = np.column_stack((n_scaled, mu_scaled, sigma_scaled, cv_scaled))
        
        # Объединяем 4 массива в одну матрицу
        self.y = self.data['best_beta'].values
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_tensor = torch.tensor([self.y[idx]], dtype=torch.float32)
        return x_tensor, y_tensor