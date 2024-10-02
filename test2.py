import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms
# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

def load_image(image_path):
    # Load an image from file
    image = Image.open(image_path).convert("RGB")
    return image
# Apply NMS manually after predictions
def apply_nms(predictions, iou_threshold=0.5):
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    
    # Perform Non-Maximum Suppression
    keep = nms(boxes, scores, iou_threshold)
    
    # Filter out the boxes, scores, and labels that survived the NMS
    predictions[0]['boxes'] = boxes[keep]
    predictions[0]['scores'] = scores[keep]
    predictions[0]['labels'] = predictions[0]['labels'][keep]
    
    return predictions
def transform_image(image):
    # Transform the image to a tensor and normalize
    image_tensor = F.to_tensor(image)
    return image_tensor

def predict(image, model):
    # Add a batch dimension as the model expects a list of images
    with torch.no_grad():
        prediction = model([image])
    return prediction

# Load and transform your image
image_path = '/home/kaleb/Real-Time-Parking-Availability/maxresdefault.jpg'  # Replace with your image path
image = load_image(image_path)
image_tensor = transform_image(image)

# Run inference
prediction = predict(image_tensor, model)

import numpy as np

# Function to plot the bounding boxes on the image with smaller label font size
def plot_predictions(image, predictions, threshold=0.5):
    # Convert the image from PIL to NumPy array for plotting
    np_image = np.array(image)
    
    # Create a plot
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(np_image)

    # Extract boxes, labels, and scores from the predictions
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    # Filter out low-confidence predictions
    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold:
            # Draw the bounding box
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            # Optionally, add label and score with smaller font size
            # ax.text(xmin, ymin, f'Car: {score:.2f}', color='white', fontsize=8,  # Smaller font size
            #         backgroundcolor='red', verticalalignment='top', bbox={'pad': 0.5})

    plt.axis('off')  # Hide axis
    # Save the output instead of showing it
    plt.savefig('output_with_boxes.jpg', bbox_inches='tight', pad_inches=0)  # Save to a file
    plt.close()  # Close the plot to avoid memory issues

prediction = apply_nms(prediction, iou_threshold=0.4)

# Plot the predictions with a confidence threshold (e.g., 0.5)
plot_predictions(image, prediction, threshold=0.2)
