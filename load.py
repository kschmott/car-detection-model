import fiftyone
from fiftyone import ViewField as F
available_datasets = fiftyone.zoo.list_zoo_datasets()
print(available_datasets)
fiftyone.config.dataset_zoo_dir = "/home/kaleb/yolov3-only-vehicles"
# dataset = fiftyone.zoo.load_zoo_dataset(
#     "coco-2017",
#     split="train",

#            label_types=["detections"],
#     label_field="detections",
#     classes=["truck", "car", "bus", "motorcycle"],
# )
# dataset = fiftyone.zoo.load_zoo_dataset(
#     "coco-2017",
#     split="validation",
#       label_types=["detections"],
#     label_field="detections",
#     classes=["truck", "car", "bus", "motorcycle"],
# )
