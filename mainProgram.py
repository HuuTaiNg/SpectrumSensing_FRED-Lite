import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir)])
        self.label_paths = sorted([os.path.join(label_dir, lbl) for lbl in os.listdir(label_dir)])

        # Define the RGB colors for each class
        self.class_colors = {
            (2, 0, 0): 0,       # LTE class
            (127, 0, 0): 1,     # NR class
            (200, 161, 159): 2  # Noise class
        }
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load label
        label = cv2.imread(self.label_paths[idx])
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        # Map RGB colors to class indices
        label_mask = np.zeros(label.shape[:2], dtype=np.uint8)
        for rgb, idx in self.class_colors.items():
            label_mask[np.all(label == rgb, axis=-1)] = idx

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            label_mask = torch.from_numpy(label_mask).long()

        return image, label_mask

# Usage example:
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # Resize to desired input size
    transforms.ToTensor()
])

train_dataset = SemanticSegmentationDataset(
    image_dir='C:\\Users\\Tai\\Desktop\\Tai\\C02_dataset\\train\\input',
    label_dir='C:\\Users\\Tai\\Desktop\\Tai\\C02_dataset\\train\\label',
    transform=train_transform
)

val_dataset = SemanticSegmentationDataset(
    image_dir='C:\\Users\\Tai\\Desktop\\Tai\\C02_dataset\\test\\input',
    label_dir='C:\\Users\\Tai\\Desktop\\Tai\\C02_dataset\\test\\label',
    transform=train_transform
)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

import torch
from torch.optim import Adam
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        # FEM
        self.fem_conv1 = nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=2)
        self.fem_conv2 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)

        self.fem_7x7conv1 = nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3)
        self.fem_5x5conv1 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.fem_3x3conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fem_1x1conv1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)

        self.fem_7x7conv2 = nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3)
        self.fem_5x5conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.fem_3x3conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fem_1x1conv2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)

        self.fem_7x7conv3 = nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3)
        self.fem_5x5conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.fem_3x3conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fem_1x1conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.fem_batchnorm = nn.BatchNorm2d(128)
        # ===============================================

        self.conv11 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv12 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv13 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv15 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)

        self.conv21 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv23 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv24 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv25 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv31 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv32 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv33 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv34 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv35 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)

        self.avgpooling = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        # ========================================
        self.bn_decoder2 = nn.BatchNorm2d(256)
        self.bn_decoder3 = nn.BatchNorm2d(256)
        
        self.conv1_de_1 = nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2)
        self.de_11 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)

        self.conv2_de_1 = nn.Conv2d(256, 64, kernel_size=5, stride=1, padding=2)
        self.de_21 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)

        self.conv3_de_1 = nn.Conv2d(256, 64, kernel_size=5, stride=1, padding=2)
        self.de_31 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)

        self.conv1_de_2 = nn.Conv2d(64, 16, kernel_size=5, stride=1, padding=2)
        self.de_12 = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0)

        self.conv2_de_2 = nn.Conv2d(64, 16, kernel_size=5, stride=1, padding=2)
        self.de_22 = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0)

        self.conv3_de_2 = nn.Conv2d(64, 16, kernel_size=5, stride=1, padding=2)
        self.de_32 = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0)

        self.conv1_de_3 = nn.Conv2d(16, 3, kernel_size=5, stride=1, padding=2)
        self.de_13 = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)

        self.conv2_de_3 = nn.Conv2d(16, 3, kernel_size=5, stride=1, padding=2)
        self.de_23 = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)


        self.conv3_de_3 = nn.Conv2d(16, 3, kernel_size=5, stride=1, padding=2)
        self.de_33 = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)


        self.conv_decoder = nn.Conv2d(9, 3, kernel_size=5, stride=1, padding=2)
        self.bn = nn.BatchNorm2d(9)

    def forward(self, x):
        # FEM
        x_ = self.fem_conv1(x)
        x_ = F.relu(x_)
        x_ = self.fem_conv2(x_)
        x = F.relu(x_)
        fe1, fe2, fe3, fe4 = torch.chunk(x, chunks=4, dim=1) 
        fe1 = F.relu(self.fem_1x1conv1(fe1))
        fe2 = F.relu(self.fem_3x3conv1(fe2))
        fe3 = F.relu(self.fem_5x5conv1(fe3))
        fe4 = F.relu(self.fem_7x7conv1(fe4))

        fe1 = F.relu(self.fem_1x1conv2(fe1))
        fe2 = F.relu(self.fem_3x3conv2(fe2))
        fe3 = F.relu(self.fem_5x5conv2(fe3))
        fe4 = F.relu(self.fem_7x7conv2(fe4))

        fe1 = F.relu(self.fem_1x1conv3(fe1))
        fe2 = F.relu(self.fem_3x3conv3(fe2))
        fe3 = F.relu(self.fem_5x5conv3(fe3))
        fe4 = F.relu(self.fem_7x7conv3(fe4))

        fem = torch.cat([fe1, fe2, fe3, fe4], dim=1)
        fem = F.relu(self.fem_batchnorm(fem))

        # ==================================================
        x1= fem
        x2 = self.avgpooling(x1)
        x3 = self.avgpooling(x2)

        x2 = F.relu(self.conv21(x2) + self.avgpooling(x1))
        x3 =  F.relu(self.conv31(x3) + self.avgpooling(self.avgpooling(x1)))
        x1 = F.relu(self.conv11(x1))

        x2 = F.relu(self.conv22(x2) + self.avgpooling(x1))
        x3 =  F.relu(self.conv32(x3) + self.avgpooling(self.avgpooling(x1)))
        x1 = F.relu(self.conv12(x1))

        x2 = F.relu(self.conv23(x2) + self.avgpooling(x1))
        x3 =  F.relu(self.conv33(x3) + self.avgpooling(self.avgpooling(x1)))
        x1 = F.relu(self.conv13(x1))

        x2 = F.relu(self.conv24(x2) + self.avgpooling(x1))
        x3 =  F.relu(self.conv34(x3) + self.avgpooling(self.avgpooling(x1)))
        x1 = F.relu(self.conv14(x1))

        x2 = F.relu(self.conv25(x2) + self.avgpooling(x1))
        x3 =  F.relu(self.conv35(x3) + self.avgpooling(self.avgpooling(x1)))
        x1 = F.relu(self.conv15(x1))   

        # ===============================================
        x2 = torch.cat([self.upsample(x2), x1], dim=1)
        x3 = torch.cat([self.upsample(self.upsample(x3)), x1], dim=1)

        x2 = self.bn_decoder2(x2)
        x3 = self.bn_decoder3(x3)

        x1_1 = F.relu(self.de_11(x1))
        x1 = self.conv1_de_1(x1)
        x1 = F.relu(x1)
        x2_1 = F.relu(self.de_21(x2))
        x2 = self.conv2_de_1(x2)
        x2 = F.relu(x2)
        x3_1 = F.relu(self.de_31(x3))
        x3 = self.conv3_de_1(x3)
        x3 = F.relu(x3)

        x1_2 = F.relu(self.de_12(x1))
        x1 = self.conv1_de_2(x1 + x1_1)
        x1 = F.relu(x1)
        x2_2 = F.relu(self.de_22(x2))
        x2 = self.conv2_de_2(x2 + x2_1)
        x2 = F.relu(x2)
        x3_2 = F.relu(self.de_32(x3))
        x3 = self.conv3_de_2(x3 + x3_1)
        x3 = F.relu(x3)

        x1_3 = F.relu(self.de_13(x1))
        x1 = self.conv1_de_3(x1 + x1_2)
        x1 = F.relu(x1) + x1_3
        x2_3 = F.relu(self.de_23(x2))
        x2 = self.conv2_de_3(x2 + x2_2)
        x2 = F.relu(x2) + x2_3
        x3_3 = F.relu(self.de_33(x3))
        x3 = self.conv3_de_3(x3 + x3_2)
        x3 = F.relu(x3) + x3_3

        x1 = torch.cat([x2, x3, x1],dim=1)
        x1 = F.relu(self.bn(x1))
        x1 = self.conv_decoder(x1)
        
        return F.softmax(x1)


model = myModel()
model = nn.DataParallel(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

def count_parameters(model):  # The function to count total parameters stored in memory, using for model
    return sum(p.numel() for p in model.parameters())

total_params = count_parameters(model)
print(f"Total parameters: {total_params}")


import time

def measure_inference_time(model, device, num_runs=100):
    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    model.eval()
    
    start_time = time.time()

    for _ in range(num_runs):
        with torch.no_grad():  
            output = model(input_tensor)
    
    end_time = time.time()
    total_time = (end_time - start_time)*1000
    avg_time_per_run = total_time / num_runs
    print(f"Total time for {num_runs} runs: {total_time:.4f} ms")
    print(f"Average time per run: {avg_time_per_run:.4f} ms")

measure_inference_time(model, device)


from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score, MulticlassAccuracy, MulticlassPrecision



def train_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0
    
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    # Instantiate metrics
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes).to(device)
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes).to(device)
    
    pbar = tqdm(dataloader, desc='Training', unit='batch')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        preds = torch.argmax(outputs, dim=1)
        
        # Update confusion matrix
        confmat(preds, labels)
        
        # Update metrics
        iou_metric(preds, labels)
        f1_metric(preds, labels)
        accuracy_metric(preds, labels)
        precision_metric(preds, labels)
        
        # Update tqdm description with metrics
        pbar.set_postfix({
            'Batch Loss': f'{loss.item():.4f}',
            'Mean Accuracy': f'{accuracy_metric.compute():.4f}',
            'Mean IoU': f'{iou_metric.compute():.4f}',
            'Mean F1 Score': f'{f1_metric.compute():.4f}',
            'Mean Precision': f'{precision_metric.compute():.4f}'
        })
    
    cm = confmat.compute().cpu().numpy()  # converting to numpy
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  
    confmat.reset()
   
    epoch_loss = running_loss / len(dataloader.dataset)
    # Calculate mean metrics
    mean_iou = iou_metric.compute().cpu().numpy()
    mean_f1 = f1_metric.compute().cpu().numpy()
    mean_accuracy = accuracy_metric.compute().cpu().numpy()
    mean_precision = precision_metric.compute().cpu().numpy()
   
    return cm_normalized, epoch_loss, mean_iou, mean_f1, mean_accuracy, mean_precision

def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    
    # Instantiate metrics
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes).to(device)
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes).to(device)


    pbar = tqdm(dataloader, desc='Evaluating', unit='batch')

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).float()
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)

            # Update confusion matrix
            confmat(preds, labels)

            # Update metrics
            iou_metric(preds, labels)
            f1_metric(preds, labels)
            accuracy_metric(preds, labels)
            precision_metric(preds, labels)


            # Update tqdm description with metrics
            pbar.set_postfix({
                'Batch Loss': f'{loss.item():.4f}',
                'Mean Accuracy': f'{accuracy_metric.compute():.4f}',
                'Mean IoU': f'{iou_metric.compute():.4f}',
                'Mean F1 Score': f'{f1_metric.compute():.4f}',
                'Mean Precision': f'{precision_metric.compute():.4f}'
            })
    
    epoch_loss = running_loss / len(dataloader.dataset)
    cm = confmat.compute().cpu().numpy()  # Convert to numpy for easy usage
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize
    confmat.reset()

    # Calculate mean metrics
    mean_iou = iou_metric.compute().cpu().numpy()
    mean_f1 = f1_metric.compute().cpu().numpy()
    mean_accuracy = accuracy_metric.compute().cpu().numpy()
    mean_precision = precision_metric.compute().cpu().numpy()

    return cm_normalized, epoch_loss, mean_iou, mean_f1, mean_accuracy, mean_precision

import copy

# Training model
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-5)
num_epochs = 50
num_classes = 3
epoch_saved = 0

best_val_accuracy = 0.0
best_model_state = None

for epoch in range(num_epochs):
    _, epoch_loss_train, iou_score_avg_train, f1_score_avg_train, accuracy_avg_train, precision_avg_train = train_epoch(model, train_dataloader, criterion, optimizer, device, num_classes)
    _, epoch_loss_val, iou_score_avg_val, f1_score_avg_val, accuracy_avg_val, precision_avg_val = evaluate(model, val_dataloader, criterion, device, num_classes)
    
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {epoch_loss_train:.4f}, Mean Accuracy: {accuracy_avg_train:.4f}, Mean IoU: {iou_score_avg_train:.4f}, Mean F1 Score: {f1_score_avg_train:.4f}, Mean Precision: {precision_avg_train:.4f}")
    print(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {accuracy_avg_val:.4f}, Mean IoU: {iou_score_avg_val:.4f}, Mean F1 Score: {f1_score_avg_val:.4f}, Mean Precision: {precision_avg_val:.4f}")
    f = open('training.txt', 'a')
    f.write(f"Epoch {epoch + 1}/{num_epochs}\n")
    f.write(f"Train Loss: {epoch_loss_train:.4f}, Mean Accuracy: {accuracy_avg_train:.4f}, mIoU: {iou_score_avg_train:.4f}, Mean F1 Score: {f1_score_avg_train:.4f}, Mean Precision: {precision_avg_train:.4f}\n")
    f.write(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {accuracy_avg_val:.4f}, mIoU: {iou_score_avg_val:.4f}, Mean F1 Score: {f1_score_avg_val:.4f}, Mean Precision: {precision_avg_val:.4f}\n")
    f.close()
    if accuracy_avg_val >= best_val_accuracy:
        epoch_saved = epoch + 1 
        best_val_accuracy = accuracy_avg_val
        best_model_state = copy.deepcopy(model.state_dict())
    
print("===================")
print(f"Best Model at epoch : {epoch_saved}")
f = open('training.txt', 'a')
f.write("===================\n")
f.write(f"Best Model at epoch : {epoch_saved}\n")
f.close()
torch.save(best_model_state, "FRED-Lite.pth")