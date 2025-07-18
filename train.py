import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os
from sklearn.metrics import classification_report

# Import các thành phần từ các file khác trong project
from src.models.model import PowerfulCNN
from src.data.datasets import get_dataloaders
from src.engine.trainer import train_one_epoch, evaluate_model
from src.utils.plots import save_loss_curves, save_confusion_matrix

def main():
    # --- 1. CẤU HÌNH ---
    # Các tham số có thể đưa vào file config (YAML, JSON) để chuyên nghiệp hơn
    IMAGE_SIZE = 512 # Bắt đầu với 224x224 cho nhanh, có thể tăng lên 512 sau
    BATCH_SIZE = 16
    NUM_EPOCHS = 25
    LEARNING_RATE = 0.001
    
    TRAIN_DIR = "data/processed/train"
    TEST_DIR = "data/processed/test"
    OUTPUT_DIR = "outputs"
    
    # Tạo thư mục output nếu chưa có
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Chọn device (GPU nếu có, không thì CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng device: {device}")

    # --- 2. CHUẨN BỊ DỮ LIỆU ---
    print("\nĐang tải dữ liệu...")
    train_loader, test_loader, class_names = get_dataloaders(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )
    print(f"Các lớp tìm thấy: {class_names}")
    print(f"Số lượng batch trong tập train: {len(train_loader)}")
    print(f"Số lượng batch trong tập test: {len(test_loader)}")

    # --- 3. KHỞI TẠO MÔ HÌNH, LOSS, OPTIMIZER ---
    print("\nĐang khởi tạo mô hình...")
    model = PowerfulCNN(num_classes=len(class_names)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Scheduler để giảm learning rate theo thời gian, giúp hội tụ tốt hơn
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --- 4. VÒNG LẶP HUẤN LUYỆN ---
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 20)
        
        # Huấn luyện
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Đánh giá trên tập test (đóng vai trò validation set ở đây)
        val_loss, val_acc, _, _ = evaluate_model(model, test_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Cập nhật scheduler
        scheduler.step()
        
        # Lưu lại mô hình có validation accuracy tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"Đã lưu mô hình tốt nhất với accuracy: {best_val_acc:.4f}")
            
    print("\n--- HUẤN LUYỆN HOÀN TẤT ---")

    # --- 5. ĐÁNH GIÁ CUỐI CÙNG VÀ BÁO CÁO ---
    print("\n--- BẮT ĐẦU ĐÁNH GIÁ CUỐI CÙNG ---")
    # Tải lại trọng số tốt nhất đã lưu
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth')))
    
    # Đánh giá lần cuối trên tập test
    final_loss, final_acc, y_true, y_pred = evaluate_model(model, test_loader, criterion, device)
    print(f"\nKết quả cuối cùng trên tập test:")
    print(f"  - Loss: {final_loss:.4f}")
    print(f"  - Accuracy: {final_acc:.4f}")
    
    # Tạo và lưu báo cáo classification
    report = classification_report(y_true, y_pred, target_names=class_names)
    report_path = os.path.join(OUTPUT_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nBáo cáo chi tiết đã được lưu tại: {report_path}")
    print(report)
    
    # Vẽ và lưu các biểu đồ
    save_loss_curves(history['train_loss'], history['val_loss'], os.path.join(OUTPUT_DIR, 'loss_curves.png'))
    save_confusion_matrix(y_true, y_pred, class_names, os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    
    print("\nQuy trình hoàn tất!")

if __name__ == '__main__':
    main() 