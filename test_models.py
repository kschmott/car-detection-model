import torch
import time
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fcos_resnet50_fpn,
    keypointrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
    ssd300_vgg16,
    ssdlite320_mobilenet_v3_large
)
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torchvision.ops import nms
# Function to load and preprocess the image
def load_and_transform_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image)
    return image, image_tensor

# Function to time the inference for a given model
def time_inference(model, image_tensor, device):
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    model.eval()  # Set the model to evaluation mode

    start_time = time.time()
    with torch.no_grad():
        prediction = model([image_tensor])
    end_time = time.time()

    inference_time = end_time - start_time
    return prediction, inference_time

# Function to plot the predictions on the image and save the output
def plot_predictions(image, predictions, model_name, threshold=0.5):
    np_image = np.array(image)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(np_image)

    # Extract boxes, labels, and scores from predictions
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    # Filter based on confidence threshold
    for box, score in zip(boxes, scores):
        if score >= threshold:
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin, f'{score:.2f}', color='white', fontsize=8, backgroundcolor='red', bbox={'pad': 0.5})

    plt.axis('off')
    plt.savefig(f'{model_name}_output.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

# List of models to test
models = {
    "Faster R-CNN": fasterrcnn_resnet50_fpn,
    "FCOS": fcos_resnet50_fpn,
    "Keypoint R-CNN": keypointrcnn_resnet50_fpn,
    "Mask R-CNN": maskrcnn_resnet50_fpn,
    "RetinaNet": retinanet_resnet50_fpn,
    "SSD300 VGG16": ssd300_vgg16,
    "SSDLite320 MobileNet V3": ssdlite320_mobilenet_v3_large,
}

# Load image
image_path = "/home/kaleb/yolov3-only-vehicles/7bce5fc2c9dee31f213cceef410ec679.jpg"  # Replace with your image path
image, image_tensor = load_and_transform_image(image_path)

# Choose device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def apply_nms(predictions, iou_threshold=0.5):
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    
    # Perform Non-Maximum Suppression
    keep = nms(boxes, scores, iou_threshold)
    
    # Filter out the boxes, scores, and labels that survived NMS
    predictions[0]['boxes'] = boxes[keep]
    predictions[0]['scores'] = scores[keep]
    predictions[0]['labels'] = predictions[0]['labels'][keep]
    
    return predictions
# Test each model
for model_name, model_func in models.items():
    print(f"Testing {model_name}...")
    model = model_func(weights="DEFAULT")
    prediction, inference_time = time_inference(model, image_tensor, device)
    apply_nms(prediction, iou_threshold=0.5)
    print(f"{model_name} inference time: {inference_time:.4f} seconds")
    
    # Plot and save the output image with bounding boxes
    plot_predictions(image, prediction, model_name, threshold=0.5)

