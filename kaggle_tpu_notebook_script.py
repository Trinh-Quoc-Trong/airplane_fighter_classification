# ==============================================================================
# CELL 1: CÀI ĐẶT THƯ VIỆN XLA
# ==============================================================================
# Lệnh này chỉ cần chạy một lần duy nhất trên Kaggle Notebook
# !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
# !python pytorch-xla-env-setup.py --version 2.0.0 --apt-packages libomp5 libopenblas-dev

# ==============================================================================
# CELL 2: IMPORTS
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

# Imports cho PyTorch/XLA
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

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
# CELL 3: ĐỊNH NGHĨA KIẾN TRÚC MODEL (Giữ nguyên)
# ==============================================================================
class PowerfulCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PowerfulCNN, self).__init__()
        # ... (Toàn bộ kiến trúc model giữ nguyên như cũ)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2))
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2))
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2, 2))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256*7*7, 1024), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(1024, 512), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.conv_block1(x); x = self.conv_block2(x); x = self.conv_block3(x); x = self.conv_block4(x)
        x = self.adaptive_pool(x); x = self.classifier(x)
        return x

# ==============================================================================
# CELL 4: CÁC HÀM TIỆN ÍCH (Vẽ biểu đồ,...)
# ==============================================================================
def save_loss_curves(train_loss, val_loss, save_path):
    plt.figure(figsize=(10, 6)); plt.plot(train_loss, label='Training Loss'); plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss Curves'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.savefig(save_path); plt.show(); plt.close()
    print(f"Biểu đồ loss đã được lưu tại: {save_path}")

def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix'); plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.savefig(save_path); plt.show(); plt.close()
    print(f"Ma trận nhầm lẫn đã được lưu tại: {save_path}")

# ==============================================================================
# CELL 5: HÀM CHÍNH ĐỂ HUẤN LUYỆN (map_fn)
# ==============================================================================
# Các cấu hình sẽ được truyền vào qua biến `flags`
FLAGS = {
    'IMAGE_SIZE': 512,
    'BATCH_SIZE': 16, # Batch size trên mỗi lõi TPU
    'NUM_EPOCHS': 25,
    'LEARNING_RATE': 0.001 * 8, # Learning rate được scale lên theo số lõi
    'DATA_DIR': "/kaggle/input/airplane-fighter-dataset/processed",
    'OUTPUT_DIR': "/kaggle/working/",
}

def map_fn(rank, flags):
    # `rank` là id của tiến trình hiện tại (từ 0 đến 7)
    
    # --- 1. THIẾT LẬP MÔI TRƯỜNG ---
    torch.manual_seed(42) # Đảm bảo tính nhất quán
    device = xm.xla_device() # Lấy thiết bị TPU cho tiến trình này
    
    # --- 2. CHUẨN BỊ DỮ LIỆU ---
    # Data Augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((flags['IMAGE_SIZE'], flags['IMAGE_SIZE'])),
            transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.1, 0.1), transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val': transforms.Compose([
            transforms.Resize((flags['IMAGE_SIZE'], flags['IMAGE_SIZE'])), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }

    train_dataset = datasets.ImageFolder(os.path.join(flags['DATA_DIR'], 'train'), data_transforms['train'])
    test_dataset = datasets.ImageFolder(os.path.join(flags['DATA_DIR'], 'test'), data_transforms['val'])
    class_names = train_dataset.classes
    
    # DistributedSampler để chia dữ liệu cho các lõi
    train_sampler = DistributedSampler(train_dataset, num_replicas=xm.xrt_world_size(), rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=xm.xrt_world_size(), rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=flags['BATCH_SIZE'], sampler=train_sampler, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=flags['BATCH_SIZE'], sampler=test_sampler, num_workers=2)

    # --- 3. KHỞI TẠO MÔ HÌNH VÀ CÁC THÀNH PHẦN KHÁC ---
    model = PowerfulCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=flags['LEARNING_RATE'])
    
    if rank == 0: # Chỉ tiến trình chính in thông tin
        print(f"Mô hình được chuyển đến device: {device}")
        print(f"Tổng số thiết bị TPU: {xm.xrt_world_size()}")
        print(f"Learning rate hiệu dụng: {flags['LEARNING_RATE']}")

    # --- 4. VÒNG LẶP HUẤN LUYỆN ---
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(flags['NUM_EPOCHS']):
        # ParallelLoader để tải dữ liệu song song trên các lõi
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        
        # Train one epoch
        model.train()
        train_loss, train_acc = 0.0, 0.0
        # ... (logic train_one_epoch gộp vào đây)
        progress_bar = tqdm(para_train_loader, desc=f"Epoch {epoch+1} Training", unit="batch", disable=(rank!=0))
        for inputs, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            xm.optimizer_step(optimizer) # << Bước quan trọng của XLA
            train_loss += loss.item()

        # Đánh giá trên tập test
        para_test_loader = pl.ParallelLoader(test_loader, [device]).per_device_loader(device)
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        correct_predictions, total_samples = 0, 0
        
        for inputs, labels in para_test_loader:
             with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels.data)
                total_samples += labels.size(0)

        # Đồng bộ hóa và tính toán metric từ tất cả các lõi
        # Sử dụng `xm.all_reduce` để cộng tổng loss/accuracy từ các lõi
        avg_train_loss = xm.mesh_reduce('train_loss_reduce', train_loss, np.sum) / len(train_dataset)
        avg_val_loss = xm.mesh_reduce('val_loss_reduce', val_loss, np.sum) / len(test_dataset)
        avg_val_acc = xm.mesh_reduce('val_acc_reduce', correct_predictions, np.sum) / len(test_dataset)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        # Chỉ tiến trình chính (rank 0) in và lưu model
        if rank == 0:
            print(f"Epoch {epoch+1} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                # Lưu model đúng cách với XLA
                xm.save(model.state_dict(), os.path.join(flags['OUTPUT_DIR'], 'best_model.pth'))
                print(f"Đã lưu mô hình tốt nhất với accuracy: {best_val_acc:.4f}")
    
    # --- 5. ĐÁNH GIÁ CUỐI CÙNG ---
    if rank == 0:
        # Tải lại mô hình tốt nhất
        model.load_state_dict(torch.load(os.path.join(flags['OUTPUT_DIR'], 'best_model.pth')))
        
        # Cần một dataloader không có sampler để đánh giá trên toàn bộ test set
        final_test_loader = DataLoader(test_dataset, batch_size=flags['BATCH_SIZE'], shuffle=False)
        
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(final_test_loader, desc="Final Evaluation"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.numpy())
                y_pred.extend(preds.cpu().numpy())

        report = classification_report(y_true, y_pred, target_names=class_names)
        print(f"\nBáo cáo chi tiết:\n{report}")
        
        # Lưu báo cáo và biểu đồ
        with open(os.path.join(flags['OUTPUT_DIR'], 'classification_report.txt'), 'w') as f: f.write(report)
        save_loss_curves(history['train_loss'], history['val_loss'], os.path.join(flags['OUTPUT_DIR'], 'loss_curves.png'))
        save_confusion_matrix(y_true, y_pred, class_names, os.path.join(flags['OUTPUT_DIR'], 'confusion_matrix.png'))
        print(f"\nQuy trình hoàn tất! Kết quả được lưu tại: {flags['OUTPUT_DIR']}")


# ==============================================================================
# CELL 7: ĐIỂM BẮT ĐẦU CHƯƠNG TRÌNH
# ==============================================================================
if __name__ == '__main__':
    # Bắt đầu 8 tiến trình song song, mỗi tiến trình chạy hàm map_fn
    xmp.spawn(map_fn, args=(FLAGS,), nprocs=8, start_method='fork') 