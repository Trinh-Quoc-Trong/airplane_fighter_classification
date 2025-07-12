import cv2
import json
import os

def crop_objects_from_coco(json_file, image_dir, output_dir):
    """
    Cắt các đối tượng từ dataset COCO
    """
    # Đọc file JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # Lấy thông tin images và annotations
    images = {img['id']: img for img in data['images']}
    annotations = data['annotations']
    
    for i, annotation in enumerate(annotations):
        image_id = annotation['image_id']
        bbox = annotation['bbox']  # [x, y, width, height]
        
        # Lấy thông tin hình ảnh
        image_info = images[image_id]
        image_path = os.path.join(image_dir, image_info['file_name'])
        
        # Đọc hình ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"Không thể đọc hình ảnh: {image_path}")
            continue
        
        # Cắt theo bounding box
        x, y, w, h = [int(coord) for coord in bbox]
        cropped_image = image[y:y+h, x:x+w]
        
        # Lưu hình ảnh đã cắt
        output_filename = f"object_{i}_{image_id}_{annotation['id']}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, cropped_image)
        
        print(f"Đã cắt và lưu: {output_filename}")

# Sử dụng
crop_objects_from_coco(
    json_file='data/raw/airplane/test/_annotations.coco.json',
    image_dir='data/raw/airplane/test/',
    output_dir='data/processed/cropped_objects/'
)