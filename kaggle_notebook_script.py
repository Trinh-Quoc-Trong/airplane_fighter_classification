# ==============================================================================
# CELL 1: IMPORTS VÀ CÀI ĐẶT
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

# Thiết lập style cho biểu đồ
plt.style.use('ggplot')


# ==============================================================================
# CELL 2: ĐỊNH NGHĨA KIẾN TRÚC MODEL (Nội dung từ src/models/model.py)
# ==============================================================================
class PowerfulCNN(nn.Module):
    """
    Một kiến trúc CNN sâu hơn và mạnh mẽ hơn được xây dựng từ đầu (from scratch)
    để phân loại hình ảnh. Được thiết kế cho ảnh đầu vào 512x512.
    """
    def __init__(self, num_classes=2):
        super(PowerfulCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

# ==============================================================================
# CELL 3: CÁC HÀM XỬ LÝ DỮ LIỆU (Nội dung từ src/data/datasets.py)
# ==============================================================================
def get_transforms(image_size):
    """
    Trả về một dictionary chứa các phép biến đổi cho tập train và validation/test.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return {'train': train_transforms, 'val': val_transforms}

def get_dataloaders(train_dir, test_dir, batch_size, image_size):
    """
    Tạo và trả về DataLoaders cho tập train và test.
    """
    data_transforms = get_transforms(image_size)
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['val'])
    
    # num_workers=2 là lựa chọn an toàn cho Kaggle/Colab
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    
    class_names = train_dataset.classes
    return train_loader, test_loader, class_names

# ==============================================================================
# CELL 4: CÁC HÀM VẼ BIỂU ĐỒ (Nội dung từ src/utils/plots.py)
# ==============================================================================
def save_loss_curves(train_loss, val_loss, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.show() # Hiển thị ngay trong notebook
    plt.close()
    print(f"Biểu đồ loss đã được lưu tại: {save_path}")

def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path)
    plt.show() # Hiển thị ngay trong notebook
    plt.close()
    print(f"Ma trận nhầm lẫn đã được lưu tại: {save_path}")

# ==============================================================================
# CELL 5: LOGIC HUẤN LUYỆN & ĐÁNH GIÁ (Nội dung từ src/engine/trainer.py)
# ==============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)
        progress_bar.set_postfix(loss=loss.item(), acc=f"{correct_predictions.double()/total_samples:.4f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")
    
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix(loss=loss.item(), acc=f"{correct_predictions.double()/total_samples:.4f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item(), all_labels, all_preds

# ==============================================================================
# CELL 6: SCRIPT CHÍNH ĐỂ HUẤN LUYỆN (Nội dung từ train.py đã được chỉnh sửa)
# ==============================================================================
def run_training():
    # --- 1. CẤU HÌNH ---
    # Thay đổi 'your-dataset-folder' thành tên thư mục dataset của bạn trên Kaggle
    DATA_DIR = "/kaggle/input/your-dataset-folder/processed" 
    OUTPUT_DIR = "/kaggle/working/"
    
    IMAGE_SIZE = 512
    BATCH_SIZE = 16 # Giảm batch size cho ảnh lớn 512x512 để tránh hết bộ nhớ GPU
    NUM_EPOCHS = 25
    LEARNING_RATE = 0.001
    
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng device: {device}")

    # --- 2. CHUẨN BỊ DỮ LIỆU ---
    print("\nĐang tải dữ liệu...")
    train_loader, test_loader, class_names = get_dataloaders(
        train_dir=TRAIN_DIR, test_dir=TEST_DIR, 
        batch_size=BATCH_SIZE, image_size=IMAGE_SIZE
    )
    print(f"Các lớp tìm thấy: {class_names}")

    # --- 3. KHỞI TẠO MÔ HÌNH, LOSS, OPTIMIZER ---
    print("\nĐang khởi tạo mô hình...")
    model = PowerfulCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --- 4. VÒNG LẶP HUẤN LUYỆN ---
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}\n" + "-" * 20)
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate_model(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1} -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"Đã lưu mô hình tốt nhất với accuracy: {best_val_acc:.4f}")
            
    print("\n--- HUẤN LUYỆN HOÀN TẤT ---")

    # --- 5. ĐÁNH GIÁ CUỐI CÙNG VÀ BÁO CÁO ---
    print("\n--- BẮT ĐẦU ĐÁNH GIÁ CUỐI CÙNG ---")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth')))
    _, final_acc, y_true, y_pred = evaluate_model(model, test_loader, criterion, device)
    
    print(f"\nĐộ chính xác cuối cùng trên tập test: {final_acc:.4f}")
    
    report = classification_report(y_true, y_pred, target_names=class_names)
    report_path = os.path.join(OUTPUT_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f: f.write(report)
    print(f"\nBáo cáo chi tiết:\n{report}")
    
    save_loss_curves(history['train_loss'], history['val_loss'], os.path.join(OUTPUT_DIR, 'loss_curves.png'))
    save_confusion_matrix(y_true, y_pred, class_names, os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    
    print(f"\nQuy trình hoàn tất! Tất cả kết quả đã được lưu tại thư mục: {OUTPUT_DIR}")

# --- Bắt đầu chạy toàn bộ quy trình ---
run_training() 