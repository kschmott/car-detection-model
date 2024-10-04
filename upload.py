
import glob
import json
from roboflow import Roboflow

# Initialize Roboflow client
rf = Roboflow(api_key="BZsJQzWQEsjBpdIeDVAZ")

# Directory path for images
dir_name = "/home/kaleb/yolov3-only-vehicles/coco-2017/train/images"
file_extension_type = ".jpg"

# Annotation file path and format (COCO .json)
annotation_filename = "/home/kaleb/yolov3-only-vehicles/coco-2017/train/labels.json"

# Load the COCO annotations
with open(annotation_filename, 'r') as f:
    coco_data = json.load(f)

# Define the relevant categories (cars, trucks, buses, motorcycles) and their new IDs
category_remap = {
    "car": 1,
    "truck": 2,
    "bus": 3,
    "motorcycle": 4
}

# Get original category IDs for relevant categories and map them to new IDs
original_category_ids = {cat['id']: category_remap[cat['name']] for cat in coco_data['categories'] if cat['name'] in category_remap}

# Get image IDs that contain the relevant annotations and update annotations
filtered_annotations = []
relevant_image_ids = set()

# Loop through annotations and remap category IDs if relevant
for annotation in coco_data['annotations']:
    if annotation['category_id'] in original_category_ids:
        # Remap the category_id to the new one
        annotation['category_id'] = original_category_ids[annotation['category_id']]
        filtered_annotations.append(annotation)
        relevant_image_ids.add(annotation['image_id'])

# Create a mapping from image ID to image filename (only for relevant images)
image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images'] if img['id'] in relevant_image_ids}

# Filter out the images that don't contain relevant categories
filtered_images = [img for img in coco_data['images'] if img['id'] in relevant_image_ids]

# Create the filtered COCO data
filtered_coco_data = {
    "images": filtered_images,
    "annotations": filtered_annotations,
    "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "truck"}, {"id": 3, "name": "bus"}, {"id": 4, "name": "motorcycle"}]
}

# Save the filtered annotations to a new file
filtered_annotation_file = '/home/kaleb/yolov3-only-vehicles/coco-2017/train/_annotations.coco.json'
with open(filtered_annotation_file, 'w') as f:
    json.dump(filtered_coco_data, f)

# Get the upload project from Roboflow workspace
project = rf.workspace().project("parking-6emp7")

# Iterate through the relevant images and upload them along with the filtered annotations
for image_id, image_filename in image_id_to_filename.items():
    image_path = f"{dir_name}/{image_filename}"  # Construct the full path to the image
    print(f"Uploading {image_filename}...")
    print(image_path, filtered_annotation_file)
    print(project.single_upload(
        image_path=image_path,
        annotation_path=filtered_annotation_file,  # Upload the filtered annotation file
        # optional parameters:
        # annotation_labelmap=labelmap_path,
        # split='train',
        # num_retry_uploads=0,
        # batch_name='batch_name',
        # tag_names=['tag1', 'tag2'],
        # is_prediction=False,
    ))
