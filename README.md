﻿# Phân loại Máy bay Chiến đấu vs. Máy bay Thương mại

Dự án này xây dựng một mô hình Deep Learning sử dụng mạng Nơ-ron Tích chập (Convolutional Neural Network - CNN) để phân loại hình ảnh giữa hai lớp: máy bay chiến đấu (fighter) và máy bay thương mại (airplane). Toàn bộ mô hình được xây dựng từ đầu (from scratch) sử dụng thư viện PyTorch, tập trung vào việc thể hiện các kỹ thuật cốt lõi trong Computer Vision.

![Confusion Matrix](outputs/confusion_matrix.png)

## Mục Chính

- [Tổng quan Công nghệ](#tổng-quan-công-nghệ)
- [Cấu trúc Dự án](#cấu-trúc-dự-án)
- [Phân tích Kỹ thuật Chi tiết](#phân-tích-kỹ-thuật-chi-tiết)
  - [1. Xử lý và Chuẩn bị Dữ liệu](#1-xử-lý-và-chuẩn-bị-dữ-liệu)
  - [2. Kiến trúc Mô hình (PowerfulCNN)](#2-kiến-trúc-mô-hình-powerfulcnn)
  - [3. Quy trình Huấn luyện (Training Pipeline)](#3-quy-trình-huấn-luyện-training-pipeline)
  - [4. Đánh giá và Trực quan hóa](#4-đánh-giá-và-trực-quan-hóa)
- [Hướng dẫn Cài đặt và Sử dụng](#hướng-dẫn-cài-đặt-và-sử-dụng)
  - [Yêu cầu](#yêu-cầu)
  - [Cài đặt](#cài-đặt)
  - [Huấn luyện](#huấn-luyện)
- [Kết quả](#kết-quả)


## Tổng quan Công nghệ

- **Ngôn ngữ & Framework**: Python, PyTorch.
- **Thư viện chính**:
  - `torch`, `torchvision`: Xây dựng, huấn luyện và xử lý dữ liệu cho mô hình Deep Learning.
  - `scikit-learn`: Tạo báo cáo phân loại chi tiết (precision, recall, f1-score) và ma trận nhầm lẫn.
  - `Pillow (PIL)`, `OpenCV`: Đọc và xử lý hình ảnh.
  - `matplotlib`, `seaborn`: Trực quan hóa dữ liệu (đồ thị loss, ma trận nhầm lẫn).
  - `tqdm`: Tạo thanh tiến trình (progress bar) trực quan cho quá trình huấn luyện.
- **Nền tảng**: Có khả năng chạy trên cả CPU và GPU (CUDA).

## Cấu trúc Dự án

Dự án được tổ chức theo một cấu trúc module hóa, giúp dễ dàng quản lý, bảo trì và mở rộng:

```
airplane_fighter_classification/
├── data/                    # Chứa toàn bộ dữ liệu
│   ├── interim/             # Dữ liệu được chia thành train/test nhưng chưa qua xử lý
│   ├── processed/           # Dữ liệu đã được xử lý, thay đổi kích thước và sẵn sàng cho mô hình
│   └── raw/                 # Dữ liệu gốc tải về
├── outputs/                 # Lưu trữ kết quả đầu ra
│   ├── best_model.pth       # Trọng số của mô hình có accuracy tốt nhất
│   ├── classification_report.txt # Báo cáo chi tiết về hiệu suất
│   ├── confusion_matrix.png # Ma trận nhầm lẫn
│   └── loss_curves.png      # Đồ thị loss của tập train và validation
├── src/                     # Mã nguồn chính của dự án
│   ├── data/
│   │   └── datasets.py      # Module định nghĩa DataLoader và các phép biến đổi ảnh
│   ├── engine/
│   │   └── trainer.py       # Logic cho việc huấn luyện và đánh giá mỗi epoch
│   ├── models/
│   │   └── model.py         # Định nghĩa kiến trúc của mô hình CNN
│   └── utils/
│       └── plots.py         # Các hàm tiện ích để vẽ biểu đồ
├── train.py                 # File thực thi chính để bắt đầu quá trình huấn luyện
└── requirements.txt         # Các thư viện Python cần thiết
```

## Phân tích Kỹ thuật Chi tiết

### 1. Xử lý và Chuẩn bị Dữ liệu (`src/data/datasets.py`)

- **Dataloader**: Sử dụng `torch.utils.data.DataLoader` để tạo các batch dữ liệu, giúp tối ưu hóa việc nạp dữ liệu vào bộ nhớ và GPU.
- **Image Augmentation (Tăng cường Dữ liệu)**: Các phép biến đổi (`transforms`) được áp dụng cho tập huấn luyện để tăng sự đa dạng của dữ liệu và chống overfitting:
  - `Resize`: Thay đổi kích thước tất cả ảnh về một kích thước đồng nhất (mặc định là 512x512 pixels) để phù hợp với đầu vào của mạng CNN.
  - `RandomHorizontalFlip`: Lật ảnh ngẫu nhiên theo chiều ngang.
  - `RandomRotation`: Xoay ảnh ngẫu nhiên một góc nhỏ.
  - `ToTensor`: Chuyển đổi ảnh từ định dạng PIL Image (giá trị pixel 0-255) sang PyTorch Tensor (giá trị 0.0-1.0).
  - `Normalize`: Chuẩn hóa giá trị các kênh màu của tensor ảnh với giá trị trung bình và độ lệch chuẩn cho trước. Điều này giúp mô hình hội tụ nhanh hơn.
- **Tập Test**: Chỉ áp dụng các phép biến đổi cơ bản (`Resize`, `ToTensor`, `Normalize`) để đảm bảo tính nhất quán khi đánh giá.

### 2. Kiến trúc Mô hình (PowerfulCNN) (`src/models/model.py`)

Mô hình `PowerfulCNN` được xây dựng từ đầu, bao gồm hai phần chính:

**a. Khối Trích xuất Đặc trưng (Feature Extractor):**

- Gồm 4 khối tích chập (`conv_block`) nối tiếp nhau.
- Mỗi khối thường bao gồm:
  - `nn.Conv2d`: Lớp tích chập để học các đặc trưng không gian (cạnh, góc, texture...). Kernel size là 3x3 với padding=1 để bảo toàn kích thước không gian của feature map.
  - `nn.BatchNorm2d`: Chuẩn hóa batch (Batch Normalization) được thêm vào sau mỗi lớp Conv2d. Kỹ thuật này giúp ổn định quá trình huấn luyện, giảm "internal covariate shift", và cho phép sử dụng learning rate cao hơn.
  - `nn.ReLU`: Hàm kích hoạt phi tuyến, giúp mô hình học các mối quan hệ phức tạp. `inplace=True` được sử dụng để tiết kiệm bộ nhớ.
  - `nn.MaxPool2d`: Lớp gộp (pooling) với kernel 2x2 và stride 2, làm giảm kích thước không gian của feature map đi một nửa, giúp giảm số lượng tham số và tạo ra các đặc trưng bất biến với các phép dịch chuyển nhỏ.

**b. Khối Phân loại (Classifier):**

- `nn.AdaptiveAvgPool2d((7, 7))`: Lớp gộp trung bình thích ứng. Đây là một lớp quan trọng giúp mô hình linh hoạt với các kích thước ảnh đầu vào khác nhau. Nó đảm bảo rằng đầu ra của bộ trích xuất đặc trưng luôn có kích thước cố định (256 kênh x 7x7) trước khi đưa vào lớp fully-connected.
- `nn.Flatten()`: "Làm phẳng" feature map 3D thành một vector 1D.
- `nn.Linear`: Các lớp kết nối đầy đủ (fully-connected) để thực hiện việc phân loại dựa trên các đặc trưng đã được trích xuất.
- `nn.Dropout(p=0.5)`: Kỹ thuật Dropout được sử dụng giữa các lớp Linear để chống overfitting bằng cách ngẫu nhiên "tắt" 50% các nơ-ron trong quá trình huấn luyện.

### 3. Quy trình Huấn luyện (Training Pipeline) (`train.py` & `src/engine/trainer.py`)

Quy trình huấn luyện được điều khiển bởi file `train.py` và logic chi tiết nằm trong `trainer.py`.

- **Device Handling**: Tự động phát hiện và sử dụng GPU (CUDA) nếu có, nếu không sẽ chuyển về CPU.
- **Optimizer**: Sử dụng `optim.Adam`, một thuật toán tối ưu hóa hiệu quả, kết hợp các ưu điểm của AdaGrad và RMSProp.
- **Loss Function**: `nn.CrossEntropyLoss`. Hàm mất mát này phù hợp cho bài toán phân loại đa lớp, nó kết hợp `LogSoftmax` và `NLLLoss` trong một lớp duy nhất, mang lại hiệu quả và sự ổn định về mặt số học.
- **Learning Rate Scheduler**: `optim.lr_scheduler.StepLR` được sử dụng để giảm learning rate theo một lịch trình định trước (giảm 10 lần sau mỗi 7 epochs). Việc này giúp mô hình "tinh chỉnh" trọng số ở giai đoạn cuối của quá trình huấn luyện, dễ dàng hội tụ vào điểm tối ưu hơn.
- **Epoch Loop**: Vòng lặp chính (`train.py`) điều khiển số epoch, gọi hàm `train_one_epoch` và `evaluate_model` (`trainer.py`).
- **Best Model Saving**: Sau mỗi epoch, mô hình được đánh giá trên tập validation (ở đây là tập test). Nếu accuracy của epoch hiện tại cao hơn accuracy tốt nhất trước đó, trọng số của mô hình (`model.state_dict()`) sẽ được lưu lại vào file `outputs/best_model.pth`.

### 4. Đánh giá và Trực quan hóa

- **Metrics**:
  - **Accuracy**: Tỷ lệ dự đoán đúng trên tổng số mẫu.
  - **Classification Report**: Sử dụng `sklearn.metrics.classification_report` để tính toán các chỉ số quan trọng cho mỗi lớp:
    - **Precision**: Độ chính xác (trong số các mẫu được dự đoán là một lớp, có bao nhiêu mẫu thực sự thuộc lớp đó).
    - **Recall**: Độ phủ (trong số tất cả các mẫu thực sự thuộc một lớp, mô hình nhận diện đúng được bao nhiêu).
    - **F1-Score**: Trung bình điều hòa của Precision và Recall, là một chỉ số cân bằng về hiệu suất.
- **Plots (`src/utils/plots.py`)**:
  - **Loss Curves**: Vẽ đồ thị train loss và validation loss qua từng epoch. Đây là công cụ chẩn đoán quan trọng: nếu validation loss tăng trong khi train loss giảm, đó là dấu hiệu của overfitting.
  - **Confusion Matrix**: Ma trận nhầm lẫn được tạo bằng `sklearn.metrics.confusion_matrix` và `seaborn.heatmap`. Nó cho thấy mô hình đang nhầm lẫn giữa các lớp nào, ví dụ, có bao nhiêu máy bay thương mại bị phân loại nhầm thành máy bay chiến đấu.

## Hướng dẫn Cài đặt và Sử dụng

### Yêu cầu

- Python 3.8+
- PyTorch
- CUDA (khuyến nghị để tăng tốc độ huấn luyện)

### Cài đặt

1.  **Clone repository:**
    ```bash
    git clone <URL_CUA_REPOSITORY>
    cd airplane_fighter_classification
    ```

2.  **Tạo môi trường ảo (khuyến nghị):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Windows: venv\Scripts\activate
    ```

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

### Huấn luyện

Để bắt đầu quá trình huấn luyện, chỉ cần chạy file `train.py`:

```bash
python train.py
```

- Quá trình huấn luyện sẽ bắt đầu, và bạn sẽ thấy thanh tiến trình cho mỗi epoch.
- Mô hình tốt nhất sẽ được lưu tại `outputs/best_model.pth`.
- Các biểu đồ và báo cáo sẽ được lưu trong thư mục `outputs`.

## Kết quả

Sau khi quá trình huấn luyện hoàn tất, các kết quả chính được lưu trong thư mục `outputs/`:

- `loss_curves.png`:
  ![Loss Curves](outputs/loss_curves.png)
- `classification_report.txt`: Cung cấp phân tích chi tiết về hiệu suất trên từng lớp.
- `confusion_matrix.png`: Cung cấp cái nhìn trực quan về các lỗi phân loại của mô hình.

Dựa trên các kết quả này, chúng ta có thể đánh giá hiệu suất của mô hình và xác định các hướng cải tiến tiếp theo, chẳng hạn như tinh chỉnh siêu tham số, sử dụng kiến trúc phức tạp hơn, hoặc thu thập thêm dữ liệu cho các lớp có hiệu suất thấp.
