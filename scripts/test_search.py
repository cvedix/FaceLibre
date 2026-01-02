#!/usr/bin/env python3
"""
Test script for face search API with new parameters.
Tests the /api/v1/detection/search endpoint with date range and threshold filters.
"""

import requests
import base64
import os
import sys
from datetime import datetime, timedelta

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

def test_search_with_filters(image_name, threshold=0.3, limit=50, from_date=None, to_date=None):
    """Test face search with date range and threshold filters."""
    image_path = os.path.join(IMAGE_DIR, image_name)
    
    if not os.path.exists(image_path):
        print(f"[-] Image not found: {image_path}")
        return False

    try:
        base64_img = encode_image(image_path)
        
        # Build payload with all parameters in body (not query params)
        payload = {
            "file": base64_img,
            "threshold": threshold,
            "limit": limit
        }
        
        if from_date:
            payload["from_date"] = from_date
        if to_date:
            payload["to_date"] = to_date
        
        print(f"\n[*] Testing search with {image_name}")
        print(f"    Parameters: threshold={threshold}, limit={limit}")
        if from_date:
            print(f"    from_date: {from_date}")
        if to_date:
            print(f"    to_date: {to_date}")
        
        response = requests.post(API_URL, json=payload, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                print(f"[-] No faces detected in image")
                return False
            
            for i, face in enumerate(results):
                matches = face.get("matches", [])
                print(f"    Face {i+1}: {len(matches)} matches found")
                
                for j, match in enumerate(matches[:5]):  # Show top 5
                    sim = match.get("similarity", 0)
                    camera_id = match.get("camera_id", "N/A")
                    timestamp = match.get("timestamp", "N/A")
                    print(f"      Match {j+1}: similarity={sim:.4f}, camera={camera_id}, time={timestamp}")
                
                if matches:
                    print(f"[+] SUCCESS: Found {len(matches)} matches for {image_name}")
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
    print("=" * 70)
    print("Face Search API Test - With New Parameters")
    print("=" * 70)
    print(f"API URL: {API_URL}")
    
    # Find first available test image
    test_image = None
    for img in TEST_IMAGES:
        if os.path.exists(os.path.join(IMAGE_DIR, img)):
            test_image = img
            break
    
    if not test_image:
        print("[-] No test images found!")
        return False
    
    print("\n" + "=" * 70)
    print("TEST 1: Basic search (no date filter)")
    print("=" * 70)
    test_search_with_filters(test_image, threshold=0.3, limit=10)
    
    print("\n" + "=" * 70)
    print("TEST 2: Search with date range (last 7 days)")
    print("=" * 70)
    now = datetime.now()
    from_date = (now - timedelta(days=7)).strftime("%Y-%m-%d 00:00:00")
    to_date = now.strftime("%Y-%m-%d 23:59:59")
    test_search_with_filters(test_image, threshold=0.3, limit=20, from_date=from_date, to_date=to_date)
    
    print("\n" + "=" * 70)
    print("TEST 3: Search with high threshold (0.6)")
    print("=" * 70)
    test_search_with_filters(test_image, threshold=0.6, limit=10)
    
    print("\n" + "=" * 70)
    print("TEST 4: Search with date range (last 30 days) and limit 100")
    print("=" * 70)
    from_date = (now - timedelta(days=30)).strftime("%Y-%m-%d 00:00:00")
    test_search_with_filters(test_image, threshold=0.3, limit=100, from_date=from_date, to_date=to_date)
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
