import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

plt.style.use("ggplot")

from sklearn.calibration import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
import os
import json
import time
import random
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from sklearn import model_selection, metrics
from timm.models.resnet import Bottleneck
from timm.layers import DropPath

# 修改全局变量
DATA_PATH = "brain-tumor-mri-dataset"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

IMG_SIZE = 512  # 修改为512
BATCH_SIZE = 8  
LR = 1e-4
GAMMA = 0.7
N_EPOCHS = 50

SAVE_PATH = "./working"

os.makedirs(SAVE_PATH, exist_ok=True)

label_map = {i: cls for i, cls in enumerate(CLASSES)}
with open(os.path.join(SAVE_PATH, "label_num_to_disease_map.json"), "w") as f:
    json.dump(label_map, f, indent=2)

def load_image_records(data_path, classes):
    image_records = []
    for label_id, cls in enumerate(classes):
        cls_dir = os.path.join(data_path, cls)
        try:
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_records.append({
                        "image_id": f"{cls}/{img_name}",
                        "label": label_id
                    })
        except FileNotFoundError as e:
            print(f"Error: {e}. Please check the data path.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
    return image_records

image_records = load_image_records(DATA_PATH, CLASSES)

if image_records is None:
    exit()

df = pd.DataFrame(image_records)


train_val_df, test_df = model_selection.train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df.label.values
)

train_df, valid_df = model_selection.train_test_split(
    train_val_df,
    test_size=0.25,
    random_state=42,
    stratify=train_val_df.label.values
)

class MRIDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.base_path = DATA_PATH

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            record = self.df.iloc[idx]
            img_path = os.path.join(self.base_path, record["image_id"])
            image = Image.open(img_path).convert("RGB")
            if self.transforms:
                image = self.transforms(image)
            return image, record["label"]
        except FileNotFoundError:
            print(f"Error: Image file {img_path} not found.")
            return None, None
        except Exception as e:
            print(f"An unexpected error occurred while loading image: {e}")
            return None, None

# 数据增强策略
transforms_train = transforms.Compose([
    transforms.Resize((IMG_SIZE + 128, IMG_SIZE + 128)),  
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=15, translate=(0.2, 0.2)),
    transforms.GaussianBlur(kernel_size=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transforms_valid = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据加载器
test_set = MRIDataset(test_df, transforms=transforms_valid, mode="test")
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2  # 增加workers加速数据加载
)

train_set = MRIDataset(train_df, transforms=transforms_train)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,  # 启用pin_memory加速
    prefetch_factor=1
    #,persistent_workers=True  # 启用persistent_workers加速
)
    

valid_set = MRIDataset(valid_df, transforms=transforms_valid)
valid_loader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=BATCH_SIZE*2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 随机中心点
    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(images, labels_onehot, alpha=1.0, prob=0.5):
    if np.random.rand() > prob:
        return images, labels_onehot
    indices = torch.randperm(images.size(0)).to(images.device)
    shuffled_images = images[indices]
    shuffled_labels = labels_onehot[indices]
    
    lam = np.random.beta(alpha, alpha)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    
    mixed_images = images.clone()
    mixed_images[:, :, bbx1:bbx2, bby1:bby2] = shuffled_images[:, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
    
    mixed_labels = labels_onehot * lam + shuffled_labels * (1 - lam)
    
    return mixed_images, mixed_labels
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if targets.dim() == 1:  # 如果传入的是整数标签
            targets = F.one_hot(targets, num_classes=4).float()

        probs = F.softmax(inputs, dim=1)
        log_probs = F.log_softmax(inputs, dim=1)
        
        ce_loss = - (targets * log_probs).sum(dim=1)  # (B,)
        pt = (targets * probs).sum(dim=1)  # (B,)
        
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_weight = torch.matmul(targets, self.alpha.unsqueeze(1)).squeeze()  # (B,)
            focal_loss = alpha_weight * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        scale = self.sigmoid(out).unsqueeze(2).unsqueeze(3)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        scale = self.sigmoid(x)
        return identity * scale

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction_ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class LightCBAM(nn.Module):
    def __init__(self, channels, drop_path=0.1):
        super().__init__()
        # 简化通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//16, 1),
            nn.ReLU(),
            nn.Conv2d(channels//16, channels, 1),
            nn.Sigmoid()
        )
        # 简化空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        identity = x
        # 应用通道注意力
        x = x * self.channel_att(x)
        # 生成空间注意力的输入：平均和最大池化的拼接
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_input = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        # 应用空间注意力
        spatial_attention = self.spatial_att(spatial_input)
        x = x * spatial_attention
        # 残差连接和DropPath
        return self.drop_path(x) + identity

# 模型创建函数
def create_model(pretrained=True):
    # 自定义Bottleneck类
    class CustomBottleneck(timm.models.resnet.Bottleneck):
        def __init__(self, *args, **kwargs):
            # 移除layer_num参数，改为后置初始化
            super().__init__(*args, **kwargs)
            self.cbam = None  # 初始化为空
                
        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.act1(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.act2(out)

            out = self.conv3(out)
            if self.cbam is not None:
                out = self.cbam(out)  # 仅在指定层使用CBAM
            out = self.bn3(out)

            if self.se is not None:
                out = self.se(out)

            if self.drop_path is not None:
                out = self.drop_path(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.act3(out)
            return out

    # 指定自定义Bottleneck
    model = timm.create_model(
        'resnet50d',
        pretrained=False,
        drop_path_rate=0.2,
        block=CustomBottleneck,
    )

    for i, block in enumerate(model.layer4):
        if i >= 1:  # 只在后两个block添加
            block.cbam = CBAM(block.conv3.out_channels)

    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, 4)
    )
    return model


def generate_grad_cam(model, target_layer, device, image_tensor, true_label, pred_label, save_path, image_id):
    """
    生成Grad-CAM热力图并保存可视化结果
    """
    # 注册钩子存储激活和梯度
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output.detach().cpu())
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach().cpu())
    
    # 注册前向和反向钩子
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    # 准备输入
    img_tensor = image_tensor.unsqueeze(0).to(device)
    img_tensor.requires_grad_(True)
    
    # 前向传播
    model.zero_grad()
    with torch.enable_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1)
        one_hot = torch.zeros_like(output)
        one_hot[0][pred] = 1.0
    
    # 反向传播
    output.backward(gradient=one_hot, retain_graph=True)
    
    # 获取激活和梯度
    activation = activations[0].squeeze()
    gradient = gradients[0].squeeze().mean(dim=(1,2), keepdim=True)
    
    # 计算加权激活
    weighted_activation = (gradient * activation).sum(dim=0)
    weighted_activation = F.relu(weighted_activation)
    
    # 归一化
    heatmap = weighted_activation.detach().numpy()
    # heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    
    # 调整热力图尺寸
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE)) if cv2 else \
              np.array(Image.fromarray(heatmap).resize((IMG_SIZE, IMG_SIZE)))
    heatmap = np.uint8(255 * heatmap)
    

    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    
    # 反归一化原始图像
    img = image_tensor.cpu().numpy().transpose(1,2,0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean).clip(0, 1)
    img = (img * 255).astype(np.uint8)
    
    # 叠加热力图
 
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

    
    # 绘制结果
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = [
        f"Original (True: {CLASSES[true_label]})",
        "Grad-CAM Heatmap",
        f"Overlay (Pred: {CLASSES[pred_label]})"
    ]
    
    for ax, data, title in zip(axes, [img, heatmap_colored, superimposed_img], titles):
        ax.imshow(data)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"grad_cam_{image_id}.png"), bbox_inches='tight')
    plt.close()
    
    # 移除钩子
    forward_handle.remove()
    backward_handle.remove()
    
    return heatmap


def main():
    #seed_everything(1001)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    try:
         # 创建模型（加载ImageNet预训练权重）
        model = create_model(pretrained=True)  # 使用修正后的创建函数
        model.to(device)
        
        # 重点：部分加载预训练权重
        pretrained_dict = timm.create_model('resnet50d', pretrained=True).state_dict()
        model_dict = model.state_dict()
        
        # 过滤不匹配的键
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        
        # 加载可匹配部分
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
    except Exception as e:
        print(f"Failed to create model: {e}")
        return


    model.to(device)
    print(model)
    # 计算训练集的类别分布
    class_counts = train_df['label'].value_counts().sort_index().values
    print(f"Training set class counts: {class_counts}")

    # 计算alpha（逆频率加权）
    alpha = 1.0 / class_counts
    alpha = alpha / alpha.sum()  # 归一化
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32).to(device)
    print(f"Alpha values: {alpha_tensor}")

    criterion = FocalLoss(alpha=alpha_tensor, gamma=2, reduction='mean')

    optimizer = torch.optim.AdamW([
        {'params': model.conv1.parameters(), 'lr': LR*0.05},   # 初始层
        {'params': model.layer1[:3].parameters(), 'lr': LR*0.1},  
        {'params': model.layer4.parameters(), 'lr': LR},       
        {'params': model.fc.parameters(), 'lr': LR*3}         
    ], weight_decay=1e-4)

    # 改进的调度策略
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR*3,
        epochs=N_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    # 数据集加载

    scaler = torch.cuda.amp.GradScaler()

    # 训练循环
    best_val_acc = 0.0
    patience = 10
    no_improve = 0
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for images, labels in tqdm(train_loader):
            if images is None or labels is None:
                continue
            images, labels = images.to(device), labels.to(device)
            
            # 转换标签为one-hot编码
            labels_onehot = F.one_hot(labels, num_classes=4).float().to(device)
            
            # 应用CutMix
            mixed_images, mixed_labels = cutmix_data(images, labels_onehot, alpha=1.0, prob=0.5)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(mixed_images)
                loss = criterion(outputs, mixed_labels)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 添加梯度裁剪
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            scheduler.step()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in valid_loader:
                if images is None or labels is None:
                    continue
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                if labels.dim() == 2:  # one-hot 编码转换
                    labels = torch.argmax(labels, dim=1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 计算指标
        epoch_time = time.time() - start_time
        train_loss /= len(train_set)
        val_loss /= len(valid_set)
        val_acc = 100 * correct / total
        
        
        
        # 早停机制
        if val_acc >= best_val_acc or val_acc == 100.0:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, "best_resnet50d+CBAM.pth"))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No improvement for {patience} epochs, stopping training!")
                break
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join(SAVE_PATH, f"checkpoint_{epoch+1}.pth"))

        print(f"Epoch {epoch+1}/{N_EPOCHS} | Time: {epoch_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        del mixed_images, mixed_labels, outputs
        torch.cuda.empty_cache()  
        gc.collect() 

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss', linestyle='--')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, color='green', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, "training_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()



   # 测试评估
    log_file_path = os.path.join(SAVE_PATH, 'test_metrics.log')  # 日志文件路径
    model = create_model(pretrained=False)  # 不加载ImageNet权重
    model.load_state_dict(torch.load(os.path.join(SAVE_PATH, "best_resnet50d+CBAM.pth")))
    model.to(device)
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    all_probs = []  

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)  # 计算概率
            
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())  # 收集概率

    test_acc = 100 * test_correct / test_total
    with open(log_file_path, 'w') as f:
        f.write(f"Final Test Accuracy: {test_acc:.2f}%\n\n")

        # 转换数据格式
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # 计算AUC-ROC
        y_true = label_binarize(all_labels, classes=[0, 1, 2, 3])

        # 各分类AUC计算
        auc_scores = {}
        for i, cls in enumerate(CLASSES):
            auc_scores[cls] = roc_auc_score(y_true[:, i], all_probs[:, i])

        # 宏观平均和微观平均
        macro_auc = roc_auc_score(y_true, all_probs, average='macro', multi_class='ovr')
        micro_auc = roc_auc_score(y_true, all_probs, average='micro', multi_class='ovr')

        f.write("AUC-ROC Scores:\n")
        for cls in CLASSES:
            print(f"{cls}: {auc_scores[cls]:.4f}")
        f.write(f"\nMacro Average: {macro_auc:.4f}\n")
        f.write(f"Micro Average: {micro_auc:.4f}\n\n")

        # 绘制ROC曲线
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green', 'purple']
        for i, cls in enumerate(CLASSES):
            fpr, tpr, _ = roc_curve(y_true[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{cls} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(SAVE_PATH, "roc_curve.png"))
        plt.show()


        report = metrics.classification_report(all_labels, all_preds, target_names=CLASSES)
        f.write("Classification Report:\n")
        f.write(report + "\n\n")

    # 绘制混淆矩阵
    cm = metrics.confusion_matrix(all_labels, all_preds)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(SAVE_PATH, "confusion_matrix.png"))
    plt.show()

    # 可视化部分
    # 选择目标层
    target_layer = model.layer4[-1].cbam  
    
    # 创建可视化目录
    grad_cam_dir = os.path.join(SAVE_PATH, "grad_cam")
    os.makedirs(grad_cam_dir, exist_ok=True)
    
    # 选择5个样本进行可视化
    sample_indices = random.sample(range(len(test_set)), 5)
    
    for i, idx in enumerate(sample_indices):
        try:
            image, true_label = test_set[idx]
            if image is None:
                continue
            
            # 生成预测
            with torch.no_grad():
                output = model(image.unsqueeze(0).to(device))
                pred_label = output.argmax().item()
            
            # 生成Grad-CAM
            generate_grad_cam(
                model=model,
                target_layer=target_layer,
                device=device,
                image_tensor=image,
                true_label=true_label,
                pred_label=pred_label,
                save_path=grad_cam_dir,
                image_id=f"sample_{i}"
            )
        except Exception as e:
            print(f"Error generating Grad-CAM for sample {idx}: {e}")



if __name__ == '__main__':
    main()
