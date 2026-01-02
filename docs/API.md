# FaceLibre API Documentation

## Base URL
```
http://localhost:8080
```

---

## 📌 Face Recognition Endpoints

### 1. Recognize Faces
**POST** `/api/v1/recognition/recognize`

Detect and recognize faces in an image. Automatically saves face embeddings to `tc_face_similarity` table for future search.

#### Request Body
```json
{
    "file": "<base64_encoded_image>",
    "camera": "camera_001",
    "metadata": {
        "location": "entrance",
        "custom_field": "value"
    }
}
```

#### Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | string | ✅ Yes | - | Base64 encoded image |
| `camera` | string | ❌ No | "" | Camera ID (saved to `camera_id` column) |
| `metadata` | object | ❌ No | {} | Additional metadata (saved to `attributes` column as JSON) |

#### Query Parameters (Optional)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 0 | Maximum faces to process |
| `det_prob_threshold` | float | 0.5 | Minimum detection confidence |
| `prediction_count` | int | 1 | Number of predictions per face |
| `face_plugins` | string | "" | Plugins: landmarks, age, gender, mask, pose, calculator |
| `status` | boolean | false | Include plugin versions |

#### Response
```json
{
    "result": [
        {
            "box": {
                "probability": 0.98,
                "x_min": 100,
                "y_min": 50,
                "x_max": 200,
                "y_max": 180
            },
            "subjects": [
                {
                    "subject": "John Doe",
                    "similarity": 0.95
                }
            ]
        }
    ]
}
```

#### Example
```bash
curl -X POST http://localhost:8080/api/v1/recognition/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "file": "/9j/4AAQSkZJRg...",
    "camera": "camera_001",
    "metadata": {"location": "entrance", "building": "A"}
  }'
```

---

### 2. Search Faces in History
**POST** `/api/v1/detection/search`

Search for similar faces in the `tc_face_similarity` table with optional date range and threshold filters.

#### Request Body
```json
{
    "file": "<base64_encoded_image>",
    "limit": 50,
    "threshold": 0.3,
    "from_date": "2025-01-01 00:00:00",
    "to_date": "2025-01-31 23:59:59"
}
```

#### Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | string | ✅ Yes | - | Base64 encoded image to search |
| `limit` | int | ❌ No | 50 | Maximum number of results |
| `threshold` | float | ❌ No | 0.3 | Minimum similarity threshold (0.0 - 1.0) |
| `from_date` | string | ❌ No | "" | Start date filter (Format: `YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS`) |
| `to_date` | string | ❌ No | "" | End date filter (Format: `YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS`) |

> **Note**: All timestamps use Vietnam timezone (UTC+7)

#### Response
```json
{
    "success": true,
    "results": [
        {
            "box": {
                "x": 100,
                "y": 50,
                "width": 100,
                "height": 130
            },
            "confidence": 0.98,
            "matches": [
                {
                    "id": "12345",
                    "camera_id": "camera_001",
                    "timestamp": "2025-01-15 14:30:00",
                    "similarity": 0.85,
                    "attributes": "{\"location\":\"entrance\"}",
                    "face_image": "<base64_cropped_face>"
                }
            ]
        }
    ]
}
```

#### Examples

**Basic Search (no filters):**
```bash
curl -X POST http://localhost:8080/api/v1/detection/search \
  -H "Content-Type: application/json" \
  -d '{
    "file": "/9j/4AAQSkZJRg..."
  }'
```

**Search with Date Range:**
```bash
curl -X POST http://localhost:8080/api/v1/detection/search \
  -H "Content-Type: application/json" \
  -d '{
    "file": "/9j/4AAQSkZJRg...",
    "from_date": "2025-01-01",
    "to_date": "2025-01-31",
    "limit": 100,
    "threshold": 0.5
  }'
```

**Search Last 7 Days with High Threshold:**
```bash
curl -X POST http://localhost:8080/api/v1/detection/search \
  -H "Content-Type: application/json" \
  -d '{
    "file": "/9j/4AAQSkZJRg...",
    "from_date": "2025-12-25 00:00:00",
    "to_date": "2026-01-02 23:59:59",
    "threshold": 0.6,
    "limit": 20
  }'
```

---

## 📊 Database Schema

### Table: `tc_face_similarity`
| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Auto-increment primary key |
| `face_embedding` | text | Face embedding vector (comma-separated floats) |
| `camera_id` | varchar(255) | Camera ID from request |
| `base64_image` | longtext | Base64 encoded cropped face image |
| `timestamp` | timestamp | Detection timestamp (Vietnam timezone) |
| `attributes` | varchar(4000) | JSON metadata |
| `created_at` | timestamp | Record creation time |

---

## ⚙️ Configuration

### Timezone
The API automatically sets MySQL session timezone to Vietnam (UTC+7):
```sql
SET time_zone = '+07:00'
```

### Default Values
| Setting | Value |
|---------|-------|
| Server Port | 8080 |
| Default Limit | 50 |
| Default Threshold | 0.3 |
| Timezone | UTC+7 (Vietnam) |

---

## 🔗 Other Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/recognition/faces` | Register a new face |
| `GET` | `/api/v1/recognition/faces` | List all registered faces |
| `DELETE` | `/api/v1/recognition/faces?subject=name` | Delete specific or all faces |
| `PUT` | `/api/v1/recognition/subjects/{name}` | Rename a subject |
| `POST` | `/api/v1/detection/extract` | Extract face embeddings only |
| `POST` | `/api/v1/recognition/search` | Search in registered faces library |
| `GET` | `/api/v1/recognition/info` | System information |
| `GET` | `/api/health` | Health check |
