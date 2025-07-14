# ============================================================
# 0. CÀI ĐẶT & IMPORT
# ------------------------------------------------------------
!pip install opencv-python-headless Pillow tqdm

import os, json, cv2 as cv
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import Counter

# ------------------------------------------------------------
IMG_DIR   = Path("data/raw/airplane/train")        # thư mục ảnh
JSON_FILE = Path("data/raw/airplane/train/_annotations.coco.json")
BLUR_THR  = 100.0   # ngưỡng độ biến thiên Laplacian, < ngưỡng ⇒ mờ
AREA_LOW, AREA_HIGH = 0.001, 0.8   # tỉ lệ diện tích bbox/ảnh để coi là outlier
# ============================================================

# ------------------------------------------------------------
# 1. ĐỌC FILE COCO & ÁNH XẠ image_id → info
# ------------------------------------------------------------
with open(JSON_FILE, "r") as f:
    coco = json.load(f)
images_info = {img["id"]: img for img in coco["images"]}
annos       = coco["annotations"]

# ------------------------------------------------------------
# 2. KIỂM TRA ẢNH HỎNG VÀ ẢNH MỜ
# ------------------------------------------------------------
broken_imgs, blur_imgs = [], []

for img_name in tqdm(os.listdir(IMG_DIR)):
    img_path = IMG_DIR / img_name
    img = cv.imread(str(img_path))
    if img is None:
        broken_imgs.append(img_name)
        continue

    # tính độ sắc nét (variance of Laplacian)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    lap_var = cv.Laplacian(gray, cv.CV_64F).var()
    if lap_var < BLUR_THR:
        blur_imgs.append((img_name, lap_var))

print("\n===== KẾT QUẢ FILE ẢNH =====")
print(f"Số ảnh hỏng   : {len(broken_imgs)}")
print(f"Số ảnh bị mờ  : {len(blur_imgs)} (ngưỡng {BLUR_THR})")
print("Ảnh mờ (5 ảnh đầu):", blur_imgs[:5])

# ------------------------------------------------------------
# 3. KIỂM TRA LABEL OUTLIER (bbox quá nhỏ / quá to)
# ------------------------------------------------------------
outlier_annos = []

for anno in annos:
    img_meta = images_info[anno["image_id"]]
    w_img, h_img = img_meta["width"], img_meta["height"]

    x, y, w, h = anno["bbox"]
    bbox_area  = w * h
    img_area   = w_img * h_img
    ratio      = bbox_area / img_area

    if ratio < AREA_LOW or ratio > AREA_HIGH:
        outlier_annos.append({
            "image": img_meta["file_name"],
            "anno_id": anno["id"],
            "ratio": round(ratio, 5)
        })

print("\n===== KẾT QUẢ LABEL =====")
print(f"Annotation outlier (area ratio < {AREA_LOW} hoặc > {AREA_HIGH}): {len(outlier_annos)}")
print("5 outlier đầu tiên:", outlier_annos[:5])

# ------------------------------------------------------------
# 4. THỐNG KÊ CLASS IMBALANCE (tùy chọn)
# ------------------------------------------------------------
cls_cnt = Counter(a["category_id"] for a in annos)
print("\n===== PHÂN BỐ CLASS =====")
for cid, cnt in cls_cnt.items():
    print(f"Class {cid:<2}: {cnt:>6} bbox")