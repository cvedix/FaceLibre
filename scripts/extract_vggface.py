import cv2
import os
import json
import base64
import random
import subprocess
from datetime import datetime, timedelta
import numpy as np

# Configuration
CONFIG_PATH = "/home/cvedix/FaceLibre/data/config.json"
DATASET_PATH = "/home/cvedix/FaceLibre/data/vggface2_112x112"
MODELS_DIR = "/home/cvedix/FaceLibre/models"
DETECTOR_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar_int8.onnx")
RECOGNIZER_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

DB_USER = "root"
DB_PASS = "Admin123456"
DB_NAME = "usc"
DB_TABLE = "tc_face_similarity"

def get_camera_ids():
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    return [c["uuid"] for c in config if "uuid" in c]

def encode_image_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def insert_into_db(embedding, camera_id, base64_image, timestamp):
    emb_str = ",".join(map(str, embedding))
    
    # Use mysql CLI to insert
    sql = f"INSERT INTO {DB_TABLE} (face_embedding, camera_id, base64_image, timestamp) VALUES ('{emb_str}', '{camera_id}', '{base64_image}', '{timestamp}');"
    
    cmd = ["mysql", "-h", "127.0.0.1", "-P", "3306", f"-u{DB_USER}", f"-p{DB_PASS}", "-e", sql, DB_NAME]

    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[-] MySQL Error: {e.stderr.decode()}")
        return False

def main():
    camera_ids = get_camera_ids()
    if not camera_ids:
        print("[-] No camera IDs found in config.json")
        return

    # Initialize OpenCV models
    detector = cv2.FaceDetectorYN.create(DETECTOR_MODEL, "", (320, 320))
    recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")

    count = 0
    # Walk through the dataset
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                h, w, _ = img.shape
                detector.setInputSize((w, h))
                _, faces = detector.detect(img)

                if faces is not None:
                    # Use the first face
                    face_det = faces[0]
                    aligned_face = recognizer.alignCrop(img, face_det)
                    feature = recognizer.feature(aligned_face)
                    embedding = feature[0]

                    # Pick a random camera ID
                    camera_id = random.choice(camera_ids)
                    
                    # Encode original image (or aligned?)
                    # The table seems to expect a base64 image.
                    # I'll use the original image or resized version if too large.
                    # VGGFace2 112x112 is small, so original is fine.
                    base64_img = encode_image_base64(img)

                    # Generate random timestamp within last 30 days
                    random_days = random.randint(0, 30)
                    random_hours = random.randint(0, 23)
                    random_minutes = random.randint(0, 59)
                    random_seconds = random.randint(0, 59)
                    ts = datetime.now() - timedelta(days=random_days, hours=random_hours, minutes=random_minutes, seconds=random_seconds)
                    timestamp_str = ts.strftime('%Y-%m-%d %H:%M:%S')

                    if insert_into_db(embedding, camera_id, base64_img, timestamp_str):
                        count += 1
                        if count % 10 == 0:
                            print(f"[+] Processed {count} images...")
                    
                    if count >= 500:
                        print("[*] Reached limit of 500 images.")
                        return



    print(f"[+] Finished. Total inserted: {count}")

if __name__ == "__main__":
    main()
