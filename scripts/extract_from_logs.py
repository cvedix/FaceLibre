#!/usr/bin/env python3
"""
Extract face embeddings from tc_face_recognition_log and insert into tc_face_similarity.

This script reads face images from usc_vinh.tc_face_recognition_log.request_body,
generates face embeddings using OpenCV's YuNet + SFace models, and inserts results
into usc.tc_face_similarity table for face search/matching.
"""

import cv2
import os
import json
import base64
import argparse
import random
import pymysql
import numpy as np
from datetime import datetime

# Configuration
MODELS_DIR = "/home/cvedix/FaceLibre/models"
DETECTOR_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar_int8.onnx")
RECOGNIZER_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
CONFIG_PATH = "/home/cvedix/FaceLibre/data/config.json"

# Database configuration
SOURCE_DB = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "Admin123456",
    "database": "usc_vinh"
}

DEST_DB = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "Admin123456",
    "database": "usc"
}

SOURCE_TABLE = "tc_face_recognition_log"
DEST_TABLE = "tc_face_similarity"


def get_camera_ids():
    """Get camera UUIDs from config.json"""
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        return [c["uuid"] for c in config if "uuid" in c]
    except:
        return ["default-camera"]


def decode_base64_image(base64_str):
    """Decode base64 string to OpenCV image"""
    try:
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None


def extract_embedding(img, detector, recognizer):
    """Extract face embedding from image"""
    if img is None or img.size == 0:
        return None
    
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None
    
    detector.setInputSize((w, h))
    detector.setScoreThreshold(0.5)  # Lower threshold for more detections
    detector.setNMSThreshold(0.3)
    _, faces = detector.detect(img)
    
    if faces is None or len(faces) == 0:
        return None
    
    # Use the first detected face
    face_det = faces[0]
    aligned_face = recognizer.alignCrop(img, face_det)
    feature = recognizer.feature(aligned_face)
    return feature[0]


def main():
    parser = argparse.ArgumentParser(description="Extract face embeddings from logs")
    parser.add_argument("--limit", type=int, default=100, help="Number of records to process")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for reading")
    parser.add_argument("--offset", type=int, default=0, help="Starting offset")
    args = parser.parse_args()

    # Initialize OpenCV models
    print("[*] Initializing face detection and recognition models...")
    detector = cv2.FaceDetectorYN.create(DETECTOR_MODEL, "", (320, 320))
    recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")
    
    camera_ids = get_camera_ids()
    print(f"[*] Loaded {len(camera_ids)} camera IDs")
    
    # Connect to databases
    print("[*] Connecting to databases...")
    source_conn = pymysql.connect(**SOURCE_DB, cursorclass=pymysql.cursors.DictCursor)
    dest_conn = pymysql.connect(**DEST_DB)
    
    source_cursor = source_conn.cursor()
    dest_cursor = dest_conn.cursor()
    
    processed = 0
    inserted = 0
    skipped = 0
    offset = args.offset
    
    print(f"[*] Processing up to {args.limit} records starting from offset {offset}...")
    
    try:
        while processed < args.limit:
            # Fetch batch from source
            query = f"""
                SELECT id, request_body, timestamp, client_ip, mac_address
                FROM {SOURCE_TABLE}
                WHERE request_body IS NOT NULL AND request_body != ''
                LIMIT {args.batch_size} OFFSET {offset}
            """
            source_cursor.execute(query)
            rows = source_cursor.fetchall()
            
            if not rows:
                print("[*] No more records to process")
                break
            
            for row in rows:
                if processed >= args.limit:
                    break
                
                processed += 1
                record_id = row["id"]
                request_body = row["request_body"]
                timestamp = row["timestamp"]
                client_ip = row.get("client_ip", "")
                
                # Parse JSON to get base64 image
                try:
                    data = json.loads(request_body)
                    base64_img = data.get("file", "")
                except json.JSONDecodeError:
                    base64_img = request_body if len(request_body) > 100 else ""
                
                if not base64_img:
                    skipped += 1
                    continue
                
                # Decode image
                img = decode_base64_image(base64_img)
                if img is None:
                    skipped += 1
                    continue
                
                # Extract embedding
                embedding = extract_embedding(img, detector, recognizer)
                if embedding is None:
                    skipped += 1
                    continue
                
                # Prepare data for insertion
                emb_str = ",".join(map(str, embedding))
                camera_id = random.choice(camera_ids) if camera_ids else client_ip or "unknown"
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Insert into destination
                try:
                    insert_query = f"""
                        INSERT INTO {DEST_TABLE} (face_embedding, camera_id, base64_image, timestamp)
                        VALUES (%s, %s, %s, %s)
                    """
                    dest_cursor.execute(insert_query, (emb_str, camera_id, base64_img, timestamp_str))
                    dest_conn.commit()
                    inserted += 1
                    
                    if inserted % 10 == 0:
                        print(f"[+] Processed: {processed}, Inserted: {inserted}, Skipped: {skipped}")
                        
                except pymysql.Error as e:
                    print(f"[-] MySQL Error for record {record_id}: {e}")
                    skipped += 1
            
            offset += args.batch_size
    
    finally:
        # Cleanup
        source_cursor.close()
        dest_cursor.close()
        source_conn.close()
        dest_conn.close()
    
    print(f"\n[*] Done! Processed: {processed}, Inserted: {inserted}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
