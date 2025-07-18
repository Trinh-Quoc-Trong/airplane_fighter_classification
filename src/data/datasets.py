import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- CÁC PHÉP BIẾN ĐỔI HÌNH ẢNH ---

def get_transforms(image_size):
    """
    Trả về một dictionary chứa các phép biến đổi cho tập train và validation/test.
    """
    # Augmentation cho tập train để tăng cường dữ liệu
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Biến đổi cho tập validation/test (không có augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return {'train': train_transforms, 'val': val_transforms}

# --- TẠO DATALOADER ---

def get_dataloaders(train_dir, test_dir, batch_size, image_size):
    """
    Tạo và trả về DataLoaders cho tập train và test.
    """
    data_transforms = get_transforms(image_size)
    
    # Tạo datasets
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=data_transforms['train']
    )
    
    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=data_transforms['val']
    )
    
    # Tạo dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    class_names = train_dataset.classes
    
    return train_loader, test_loader, class_names 