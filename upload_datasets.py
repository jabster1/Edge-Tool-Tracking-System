#Script to upload all the datasets and each of their image and label images for train, validate, and test folders
#Script will upload to RoboFlow overnight so that I can go to bed and wake up with the data in there
#By morning I will be able to split the data and use it through roboflow to train my YOLO model on Industrial Tools
import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

workspace = rf.workspace("jadens-workspace-mkbgq")
project = workspace.project("industrial-tools-base")

datasets = [
    "datasets/mechanical-tools-10000-3",
    "datasets/screwdriver-2",
    "datasets/tools-detection"
]

splits = ["train", "valid", "test"]

for dataset in datasets:
    for split in splits:
        images_path = f"{dataset}/{split}/images"
        labels_path = f"{dataset}/{split}/labels"
        
        if os.path.exists(images_path):
            print(f"Uploading {images_path}...")
            for image_file in os.listdir(images_path):
                if image_file.endswith((".jpg", ".jpeg", ".png")):
                    image_path = f"{images_path}/{image_file}"
                    label_file = image_file.replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt")
                    label_path = f"{labels_path}/{label_file}"
                    
                    if os.path.exists(label_path):
                        project.upload(
                            image_path=image_path,
                            annotation_path=label_path,
                            annotation_overwrite=True
                        )
            print(f"Done: {dataset}/{split}")

print("All datasets uploaded successfully.")