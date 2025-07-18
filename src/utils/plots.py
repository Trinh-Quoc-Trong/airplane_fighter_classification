import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

def save_loss_curves(train_loss, val_loss, save_path):
    """
    Vẽ và lưu biểu đồ loss của tập train và validation.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Biểu đồ loss đã được lưu tại: {save_path}")

def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Vẽ và lưu ma trận nhầm lẫn (confusion matrix).
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path)
    plt.close()
    print(f"Ma trận nhầm lẫn đã được lưu tại: {save_path}") 