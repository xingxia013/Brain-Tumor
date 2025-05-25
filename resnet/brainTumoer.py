import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import pandas as pd
from sklearn import model_selection
from PIL import Image
import random

#数据
IMG_SIZE = 224
BATCH_SIZE = 16
LR = 2e-05
GAMMA = 0.7
N_EPOCHS = 10
#预处理图片
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transforms_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transforms_valid = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


#数据路径
DataPath='E:/programming/code/resnet/archive/brain-tumor-mri-dataset'

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
image_records = []
for label_id, cls in enumerate(CLASSES):
    cls_dir = os.path.join(DataPath, cls)
    for img_name in os.listdir(cls_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_records.append({
                "image_id": f"{cls}/{img_name}",
                "label": label_id
            })

df = pd.DataFrame(image_records)
#将数据分成训练集和测试集

train_df, test_df =model_selection.train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df.label.values
)

class MRIDataset(torch.utils.data.Dataset):
    
    def __init__(self, df, transforms=None, mode="train"):
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.base_path = DataPath

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        img_path = os.path.join(self.base_path, record["image_id"])

        image = Image.open(img_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)
            
        return image, record["label"]

#train_loader =DataLoader(train_df, batch_size=32, shuffle=True)
train_dataset = MRIDataset(train_df, transforms=transforms_train)
test_dataset = MRIDataset(test_df, transforms=transforms_valid)



train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 实例化你自己的 ResNet 模型
model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=4)

#定义损失函数和优化器
criterion=nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

#将模型移动到GPU
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#训练模型

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(1001)

for epoch in range(10):
    running_loss=0.0
    for i, (inputs,labels) in enumerate(train_loader):
        inputs,labels=inputs.to(device),labels.to(device)

        optimizer.zero_grad()

        #前向传播
        outputs=model(inputs)

        loss=criterion(outputs,labels)

        #反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if(i+1)%10==0:
            print(f'Epoch [{epoch+1}], Step {i+1}, Loss: {running_loss/10}')
            running_loss=0.0

print('Finished Training')
#保存模型
torch.save(model.state_dict(),'brainTumors.pth')
#加载测试数据

#test_loader = DataLoader(valid_df, batch_size=32, shuffle=False)

#加载模型
model.load_state_dict(torch.load('brainTumors.pth'))
model.eval()
#测试模型
correct=0
total=0

with torch.no_grad():
    for inputs,labels in test_loader:
        inputs,labels=inputs.to(device),labels.to(device)
        outputs=model(inputs)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
print(f'Accuracy of the network on the test images:  {(100 * correct / total)} %')
