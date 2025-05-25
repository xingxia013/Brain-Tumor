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
#加载resnet模型
model=models.resnet18(pretrained=True)#加载预训练模型
mun_features=model.fc.in_features#提取特征
mun_classes=nn.Linear(mun_features,4)#全连接层

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

for epoch in range(3):
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
