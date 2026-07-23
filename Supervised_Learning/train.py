import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from Supervised_Learning.DroneDataset import DroneDataset
from Supervised_Learning.BetaPredictorMLP import BetaPredictorMLP

def train_model():
    
    print("Data loading...")
    dataset = DroneDataset('drone_dataset.csv')
    
    # divide the dataset into training and testing sets (80% for training, 20% for testing)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # DataLoader divides the dataset into batches and shuffles it for training
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. Инициализация нейросети
    model = BetaPredictorMLP()
    
    # 3. Setup loss function and optimizer
    # MSELoss (Mean Squared Error) - standard loss function for regression tasks
    criterion = nn.MSELoss() 
    # Adam optimizer is a popular choice for training neural networks
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    
    epochs = 300 # how many times we go through the entire dataset during training
    
    print("Начинаем обучение...")
    
    # 4. Главный цикл обучения
    for epoch in range(epochs):
        model.train() # train mode enables features like dropout and batch normalization
        running_loss = 0.0
        
        for features, targets in train_loader:
            optimizer.zero_grad()               # set gradients to zero before backpropagation
            
            outputs = model(features)           # nn predicts the best beta for the given features
            loss = criterion(outputs, targets)  # compute the loss (how far off the predictions are from the actual values)
            
            loss.backward()                     # compute gradients for each parameter based on the loss
            optimizer.step()                    # update the parameters (weights and biases) of the model based on the computed gradients
            
            running_loss += loss.item()
            
        # each 10 epochs, print the average loss for that epoch
        if (epoch + 1) % 10 == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")
            
    # 5. Сохраняем "мозг" на жесткий диск
    torch.save(model.state_dict(), 'trained_beta_predictor.pth')
    print("Training completed! Model weights saved to 'trained_beta_predictor.pth'")

if __name__ == "__main__":
    train_model()