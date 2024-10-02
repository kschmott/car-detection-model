import json
import os

# Path to your data folder and annotations
dataset_dir = "/home/kaleb/yolov3-only-vehicles/coco-2017/train"
labels_path = os.path.join(dataset_dir, "labels.json")

# Load the labels JSON file
with open(labels_path, "r") as f:
    labels = json.load(f)

# Get all image filenames from the dataset folder
dataset_images = set(os.listdir(dataset_dir))

# Extract image filenames from labels.json
annotation_images = set()
object_keys = labels.keys() 
print(object_keys)

for annotation in labels['categories']:  # Assuming COCO format
    print(annotation)

# Compare the two sets
missing_images = annotation_images - dataset_images
extra_images = dataset_images - annotation_images

# print("Missing images in dataset folder:", missing_images)
print("Extra images in dataset folder:", extra_images)
