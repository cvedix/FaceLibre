import cv2
import os
import json
import base64
import uuid
import subprocess
from datetime import datetime
import numpy as np

# Configuration
DATASET_PATH = "/home/cvedix/FaceLibre/data/vggface2_112x112"
MODELS_DIR = "/home/cvedix/FaceLibre/models"
DETECTOR_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar_int8.onnx")
RECOGNIZER_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

DB_USER = "root"
DB_PASS = "Admin123456"
DB_NAME = "usc"
DB_TABLE = "tc_face_library"

def get_mac_address():
    try:
        cmd = "cat /sys/class/net/$(ip route show default | awk '/default/ {print $5}' | head -1)/address 2>/dev/null || echo 'unknown'"
        result = subprocess.check_output(cmd, shell=True).decode().strip()
        return result
    except:
        return "unknown"

def get_machine_id():
    try:
        with open("/etc/machine-id", "r") as f:
            return f.read().strip()
    except:
        return "unknown"

def encode_image_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def insert_into_library(image_id, subject, embedding, base64_image, mac, machine):
    emb_str = ",".join(map(str, embedding))
    
    # Use mysql CLI to insert
    sql = f"INSERT INTO {DB_TABLE} (image_id, subject, embedding, base64_image, mac_address, machine_id, created_at) VALUES ('{image_id}', '{subject}', '{emb_str}', '{base64_image}', '{mac}', '{machine}', NOW());"
    
    cmd = ["mysql", "-h", "127.0.0.1", "-P", "3306", f"-u{DB_USER}", f"-p{DB_PASS}", "-e", sql, DB_NAME]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[-] MySQL Error: {e.stderr.decode()}")
        return False

def main():
    mac = get_mac_address()
    machine = get_machine_id()

    # Initialize OpenCV models
    detector = cv2.FaceDetectorYN.create(DETECTOR_MODEL, "", (320, 320))
    recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")

    count = 0
    # Walk through the dataset but only take first successful image per directory
    folders = sorted([f for f in os.listdir(DATASET_PATH) if f.startswith("id_")])
    
    for folder in folders:
        folder_path = os.path.join(DATASET_PATH, folder)
        if not os.path.isdir(folder_path):
            continue
            
        print(f"[*] Processing subject: {folder}...")
        
        # Try images in the folder until one works
        success = False
        files = sorted(os.listdir(folder_path))
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                h, w, _ = img.shape
                detector.setInputSize((w, h))
                _, faces = detector.detect(img)

                if faces is not None and len(faces) > 0:
                    # Use the first face
                    face_det = faces[0]
                    aligned_face = recognizer.alignCrop(img, face_det)
                    feature = recognizer.feature(aligned_face)
                    embedding = feature[0]

                    image_id = str(uuid.uuid4())
                    base64_img = encode_image_base64(img)

                    if insert_into_library(image_id, folder, embedding, base64_img, mac, machine):
                        count += 1
                        print(f"[+] Registered {folder}")
                        success = True
                        break # Go to next folder
        
        # Testing limit
        if count >= 1000:
            print("[*] Reached limit of 1000 subjects.")
            break


    print(f"[+] Finished. Total registered: {count}")

if __name__ == "__main__":
    main()
