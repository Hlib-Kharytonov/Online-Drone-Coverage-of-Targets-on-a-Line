import torch

weights = torch.load('trained_beta_predictor.pth')

print("Content of the .pth file:")
print("-" * 30)
for layer_name, matrix in weights.items():
    print(f"Layer: {layer_name: <15} | Matrix Size: {list(matrix.size())}")