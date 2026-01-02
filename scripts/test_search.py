#!/usr/bin/env python3
"""
Test script for face search API.
Tests the /api/v1/detection/search endpoint with known test images.
"""

import requests
import base64
import os
import sys

API_URL = "http://localhost:8080/api/v1/detection/search"
IMAGE_DIR = "/home/cvedix/FaceLibre/data"

# Test images to use for search
TEST_IMAGES = [
    "TEST_1.png",
    "TEST_3.png",
    "TEST_5.png"
]

def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_search(image_name, threshold=0.3, limit=10):
    """Test face search with a given image."""
    image_path = os.path.join(IMAGE_DIR, image_name)
    
    if not os.path.exists(image_path):
        print(f"[-] Image not found: {image_path}")
        return False

    try:
        base64_img = encode_image(image_path)
        payload = {"file": base64_img}
        params = {"threshold": threshold, "limit": limit}
        
        print(f"\n[*] Testing search with {image_name} (threshold={threshold})...")
        response = requests.post(API_URL, json=payload, params=params, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            faces = data.get("faces", [])
            
            if not faces:
                print(f"[-] No faces detected in image")
                return False
            
            for i, face in enumerate(faces):
                matches = face.get("matches", [])
                print(f"    Face {i+1}: {len(matches)} matches found")
                
                for j, match in enumerate(matches[:3]):  # Show top 3
                    sim = match.get("similarity", 0)
                    camera_id = match.get("camera_id", "N/A")
                    timestamp = match.get("timestamp", "N/A")
                    print(f"      Match {j+1}: similarity={sim:.4f}, camera={camera_id}, time={timestamp}")
                
                if matches:
                    print(f"[+] SUCCESS: Found matches for {image_name}")
                    return True
                else:
                    print(f"[-] No matches found in database for detected face")
                    return False
        else:
            print(f"[-] API Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"[-] Error: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("Face Search API Test")
    print("=" * 60)
    print(f"API URL: {API_URL}")
    
    success_count = 0
    total = len(TEST_IMAGES)
    
    for img in TEST_IMAGES:
        if test_search(img, threshold=0.3):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {success_count}/{total} images found matches")
    print("=" * 60)
    
    return success_count > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
