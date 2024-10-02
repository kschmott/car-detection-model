

conversion = {3: 0, 4: 1, 6: 2, 8:3}
import os
import json

# Define the paths for the COCO labels.json and images directory
labels_json_path = './coco-2017/train/labels.json'
images_dir = './coco-2017/train/images'
yolo_labels_dir = './coco-2017/train/labels/'

# Create YOLO labels directory if it doesn't exist
os.makedirs(yolo_labels_dir, exist_ok=True)

# Load the COCO labels.json
with open(labels_json_path) as f:
    data = json.load(f)

# List of category IDs to exclude
excluded_categories = {3, 6, 8, 4}  # Car, Bus, Truck, Motorcycle

# Loop through COCO annotations and convert to YOLOv5 format
for annotation in data['annotations']:
    category_id = annotation['category_id']

    # Skip excluded categories
    if category_id not in excluded_categories:
        continue

    image_id = annotation['image_id']
    category_id = conversion[category_id]
    # Get image info from the corresponding 'images' section
    image_info = next(img for img in data['images'] if img['id'] == image_id)
    img_width = image_info['width']
    img_height = image_info['height']
    file_name = image_info['file_name']

    # COCO bbox: [x_min, y_min, width, height]
    bbox = annotation['bbox']
    x_min, y_min, width, height = bbox

    # Convert to YOLO format
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    # YOLO annotation format: [class_id, x_center, y_center, width, height]
    yolo_annotation = [category_id, x_center, y_center, width_norm, height_norm]

    # Prepare the output YOLO label file path (.txt file)
    yolo_file_name = os.path.join(yolo_labels_dir, f"{file_name.split('.')[0]}.txt")

    # Append YOLO annotation to the corresponding label file
    with open(yolo_file_name, 'a') as yolo_file:
        yolo_file.write(' '.join([str(a) for a in yolo_annotation]) + '\n')

print(f"YOLO annotations have been saved to {yolo_labels_dir}")

# Define the paths for the COCO labels.json and images directory for the validation set
labels_json_path = './coco-2017/validation/labels.json'
images_dir = './coco-2017/validation/images'
yolo_labels_dir = './coco-2017/validation/labels/'

# Create YOLO labels directory if it doesn't exist
os.makedirs(yolo_labels_dir, exist_ok=True)

# Load the COCO labels.json
with open(labels_json_path) as f:
    data = json.load(f)

# List of category IDs to exclude
excluded_categories = {3, 6, 8, 4}  # Car, Bus, Truck, Motorcycle

# Loop through COCO annotations and convert to YOLOv5 format
for annotation in data['annotations']:
    category_id = annotation['category_id']

    # Skip excluded categories
    if category_id not in excluded_categories:
        continue

    image_id = annotation['image_id']
    category_id = conversion[category_id]
    # Get image info from the corresponding 'images' section
    image_info = next(img for img in data['images'] if img['id'] == image_id)
    img_width = image_info['width']
    img_height = image_info['height']
    file_name = image_info['file_name']

    # COCO bbox: [x_min, y_min, width, height]
    bbox = annotation['bbox']
    x_min, y_min, width, height = bbox

    # Convert to YOLO format
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    # YOLO annotation format: [class_id, x_center, y_center, width, height]
    yolo_annotation = [category_id, x_center, y_center, width_norm, height_norm]

    # Prepare the output YOLO label file path (.txt file)
    yolo_file_name = os.path.join(yolo_labels_dir, f"{file_name.split('.')[0]}.txt")

    # Append YOLO annotation to the corresponding label file
    with open(yolo_file_name, 'a') as yolo_file:
        yolo_file.write(' '.join([str(a) for a in yolo_annotation]) + '\n')

print(f"YOLO annotations for the validation set have been saved to {yolo_labels_dir}")
