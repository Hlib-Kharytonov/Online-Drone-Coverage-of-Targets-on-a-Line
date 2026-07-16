import torch
from torchvision.models import resnet18, ResNet18_Weights

# 1. Отправляем модель на видеокарту
model = resnet18(weights=ResNet18_Weights.DEFAULT).to('cuda')

# 2. Отправляем фейковую картинку и ответы на видеокарту
data = torch.rand(1, 3, 64, 64).to('cuda')
labels = torch.rand(1, 1000).to('cuda')

prediction = model(data)
loss = (prediction - labels).sum()
loss.backward()

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()

print("Шаг обучения на видеокарте прошел успешно!")