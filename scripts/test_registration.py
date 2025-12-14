import requests
import base64
import os
import sys

# Configuration
API_URL = "http://localhost:8080/api/v1/recognition/faces"
IMAGE_DIR = "./data"
IMAGES = [
    "TEST_1_Validate.png",
    "TEST_1.png", 
    "TEST_2_Validate.png",
    "TEST_2.png",
    "TEST_3_Validate.png",
    "TEST_3.png",
    "TEST_4.png",
    "TEST_5.png"
]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def register_face(image_name):
    # Subject name is the filename without extension
    subject = os.path.splitext(image_name)[0]
    image_path = os.path.join(IMAGE_DIR, image_name)
    
    if not os.path.exists(image_path):
        print(f"[-] Image not found: {image_path}")
        return False

    try:
        base64_img = encode_image(image_path)
        
        # Prepare params and body
        params = {"subject": subject}
        payload = {"file": base64_img}
        
        print(f"[*] Registering {subject}...")
        response = requests.post(API_URL, params=params, json=payload)
        
        if response.status_code == 200:
            print(f"[+] Success: {response.json()}")
            return True
        else:
            print(f"[-] Failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"[-] Error registering {subject}: {str(e)}")
        return False

def main():
    print(f"Starting registration test on {API_URL}")
    success_count = 0
    
    for img in IMAGES:
        if register_face(img):
            success_count += 1
            
    print(f"\nFinished. Successful: {success_count}/{len(IMAGES)}")

if __name__ == "__main__":
    main()
