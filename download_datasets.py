import os
from dotenv import load_dotenv
from roboflow import Roboflow

# Load API key from .env
load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise ValueError("ROBOFLOW_API_KEY not found. Check your .env file.")

rf = Roboflow(api_key=api_key)

# Dataset 1 - Mechanical Tools 10000 (~9,300 images)
print("Downloading Mechanical Tools 10000...")
proj1 = rf.workspace("mechanical-tools").project("mechanical-tools-10000")
dataset1 = proj1.version(3).download("yolov11", location="datasets/mechanical-tools-10000")

# Dataset 2 - Tools Detection (broad class variety)
print("Downloading Tools Detection...")
proj2 = rf.workspace("manual-tools").project("tools-detection-lzcwz")
dataset2 = proj2.version(1).download("yolov11", location="datasets/tools-detection-lzcwz")

# Dataset 3 - Screwdriver specific dataset
print("Downloading Screwdriver Dataset...")
proj3 = rf.workspace("screw-driver").project("screwdriver-q5rcg")
dataset3 = proj3.version(2).download("yolov11", location="datasets/screwdriver-q5rcg")

print("All datasets downloaded.")
print(f"Dataset 1 location: {dataset1.location}")
print(f"Dataset 2 location: {dataset2.location}")
print(f"Dataset 3 location: {dataset3.location}")