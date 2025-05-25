import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
train_dataset = datasets.ImageFolder(root='E:/programming/code/resnet/Brain Tumor Data Set/Brain Tumor preprocessed', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义ResNet模型
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2个类别
#model.load_state_dict(torch.load('resnet18_model1.pth'))  # 不加载预训练权重

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):  # 迭代3个epoch
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # 打印每一个step的损失
        if (i+1) % 10 == 0:  # 每10个batch打印一次
            print(f'Epoch {epoch+1}, Step {i+1}, Loss: {running_loss/10}')
            running_loss = 0.0

# 保存模型
torch.save(model.state_dict(), 'resnet18_model3.pth')

# 加载模型进行测试
model.load_state_dict(torch.load('resnet18_model3.pth'))
model.eval()

test_dataset = datasets.ImageFolder(root='E:/programming/code/resnet/Brain Tumor Data Set/Brain Tumor preprocessed', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct / total}%')
