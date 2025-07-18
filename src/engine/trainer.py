import torch
from tqdm import tqdm
import torch.nn as nn

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Thực hiện một epoch huấn luyện.
    """
    model.train()  # Đặt mô hình ở chế độ huấn luyện
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Sử dụng tqdm để hiển thị thanh tiến trình
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Xóa các gradient cũ
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass và tối ưu hóa
        loss.backward()
        optimizer.step()
        
        # Tính toán thống kê
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)

        # Cập nhật thanh tiến trình
        progress_bar.set_postfix(loss=loss.item(), acc=f"{correct_predictions.double()/total_samples:.4f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    
    return epoch_loss, epoch_acc.item()


def evaluate_model(model, dataloader, criterion, device):
    """
    Thực hiện đánh giá mô hình trên tập validation hoặc test.
    """
    model.eval()  # Đặt mô hình ở chế độ đánh giá
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")
    
    with torch.no_grad():  # Không tính toán gradient trong quá trình đánh giá
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