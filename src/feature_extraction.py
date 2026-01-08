# part 2a

"""
Feature extraction for ASL hand pose dataset using Google MediaPipe.
Extracts 21 hand landmarks (x, y, z) per image, producing a 63-dimensional
feature vector per instance.
"""
import os
import cv2
import csv
import numpy as np
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Create HandLandmarker
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
# Path to ASL images
IMAGE_DIR = ROOT / "data" / "CW2_dataset_final"
OUTPUT_CSV = ROOT / "data" / "extracted_features" / "hand_landmarks.csv"

base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# CSV Headers

header = ["instance_id"]

for i in range(21):
    header.extend([f"x{i}", f"y{i}", f"z{i}"])

header.append("label")

# Ensure that the output CSV dir is valid
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Write CSV
with open(OUTPUT_CSV, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    instance_id = 0

    # Loop through ASL labels (A–J)
    for label in sorted(os.listdir(IMAGE_DIR)):
        label_path = os.path.join(IMAGE_DIR, label)

        # Skip anything that isn't a folder
        if not os.path.isdir(label_path):
            continue

        # Loop through images in each label folder
        for filename in os.listdir(label_path):

            # Process only .jpg images
            if not filename.lower().endswith(".jpg"):
                continue

            image_path = os.path.join(label_path, filename)
            image = cv2.imread(image_path)

            if image is None:
                continue

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb
            )

            result = detector.detect(mp_image)

            # Treat failed detections as noise
            if not result.hand_landmarks:
                continue

            landmarks = result.hand_landmarks[0]

            row = [instance_id]

            for lm in landmarks:
                row.extend([lm.x, lm.y, lm.z])

            row.append(label)
            writer.writerow(row)

            instance_id += 1


print("Feature extraction complete.")
print(f"Saved to: {OUTPUT_CSV}")
