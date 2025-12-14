# FaceLibre - Face Recognition REST API

A high-performance REST API server for face registration and recognition, powered by **Drogon** framework, OpenCV YuNet (detection) and SFace (recognition).

## Features

- 🎭 **Face Registration** with data augmentation (flip, brightness, contrast)
- 🔍 **Face Recognition** with cosine similarity matching
- 📊 **Face Detection** returning bounding boxes and landmarks
- 💾 **Persistent Storage** using text-based database
- 🚀 **High-Performance** async HTTP server with Drogon framework
- 🌐 **REST API** with CORS support

## Requirements

- CMake 3.14+
- OpenCV 4.x with DNN module
- C++17 compiler (GCC 8+ or Clang 7+)

## Quick Start

### 1. Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 2. Download Models

Download model files to `models/` directory:

```bash
# Face Detection (YuNet)
wget -P models/ https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar_int8.onnx

# Face Recognition (SFace)
wget -P models/ https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

### 3. Run Server

```bash
./face_api_server 8080
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8080/api/health
```

### Register Face

```bash
curl -X POST http://localhost:8080/api/register \
  -F "image=@photo.jpg" \
  -F "name=John Doe"
```

### Recognize Faces

```bash
curl -X POST http://localhost:8080/api/recognize \
  -F "image=@photo.jpg"
```

### List Persons

```bash
curl http://localhost:8080/api/persons
```

### Delete Person

```bash
curl -X DELETE http://localhost:8080/api/persons/John%20Doe
```

### Detect Faces (no recognition)

```bash
curl -X POST http://localhost:8080/api/detect \
  -F "image=@photo.jpg"
```

## API Response Examples

### Registration Response

```json
{
  "success": true,
  "message": "Registered John Doe with 5 augmented variants",
  "person": {
    "name": "John Doe",
    "embedding_count": 5
  }
}
```

### Recognition Response

```json
{
  "success": true,
  "faces": [
    {
      "name": "John Doe",
      "similarity": 0.85,
      "confidence": 0.95,
      "box": {"x": 100, "y": 50, "width": 120, "height": 140}
    }
  ],
  "count": 1
}
```

## Project Structure

```
FaceLibre/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── main.cpp              # REST API server
│   ├── face_database.hpp     # Face database class
│   ├── face_database.cpp
│   ├── face_service.hpp      # Face detection/recognition
│   └── face_service.cpp
├── models/                   # OpenCV DNN models
│   ├── face_detection_yunet_2023mar_int8.onnx
│   └── face_recognition_sface_2021dec.onnx
├── data/
│   └── face_database.txt     # Registered faces
└── build/
```

## License

MIT License
