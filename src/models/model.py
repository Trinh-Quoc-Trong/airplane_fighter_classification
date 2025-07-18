import torch.nn as nn
import torch.nn.functional as F

class PowerfulCNN(nn.Module):
    """
    Một kiến trúc CNN sâu hơn và mạnh mẽ hơn được xây dựng từ đầu (from scratch)
    để phân loại hình ảnh. Được thiết kế cho ảnh đầu vào 512x512.
    
    Các cải tiến chính:
    - Sâu hơn: Nhiều lớp Conv2d hơn để học đặc trưng phức tạp.
    - BatchNorm2d: Giúp ổn định và tăng tốc quá trình huấn luyện.
    - AdaptiveAvgPool2d: Giúp mô hình linh hoạt với kích thước đầu vào và
      tổng hợp đặc trưng hiệu quả trước khi phân loại.
    - Tổ chức theo khối (Block): Code sạch sẽ và dễ bảo trì hơn.
    """
    def __init__(self, num_classes=2):
        super(PowerfulCNN, self).__init__()

        # --- KHỐI TÍCH CHẬP (FEATURE EXTRACTOR) ---
        # Input: 3 x 512 x 512
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32 x 256 x 256
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64 x 128 x 128
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 128 x 64 x 64
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 256 x 32 x 32
        )

        # --- BỘ PHẬN PHÂN LOẠI (CLASSIFIER) ---

        # Adaptive Pooling giúp tạo ra output có kích thước cố định (7x7)
        # bất kể kích thước feature map đầu vào.
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
        # Luồng dữ liệu qua các khối tích chập
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        # Qua lớp adaptive pooling và bộ phân loại
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        
        return x

# --- Cách sử dụng ---
# if __name__ == '__main__':
#     import torch
#     # Kiểm tra mô hình với một input giả lập
#     # Input ảnh màu (3 kênh) kích thước 512x512
#     dummy_input = torch.randn(4, 3, 512, 512) # (batch_size, channels, height, width)
#     
#     model = PowerfulCNN(num_classes=2)
#     output = model(dummy_input)
#     
#     print(f"Kiến trúc mô hình:\n{model}")
#     print(f"\nKích thước input: {dummy_input.shape}")
#     print(f"Kích thước output: {output.shape}") # Phải là (4, 2)
#     # Đếm số lượng tham số
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Tổng số tham số có thể huấn luyện: {total_params:,}")