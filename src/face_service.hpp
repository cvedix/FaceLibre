#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect/face.hpp>

#include "face_database.hpp"

/**
 * @brief Face detection and recognition service
 * 
 * Uses OpenCV YuNet for detection and SFace for recognition.
 * Supports data augmentation during registration.
 */
class FaceService {
public:
    struct FaceInfo {
        cv::Rect box;
        float confidence;
        std::vector<float> landmarks;  // 5 landmarks (x,y pairs)
        std::vector<float> embedding;
        std::string name;
        float similarity;
    };

    struct RegisterResult {
        bool success;
        std::string message;
        std::string name;
        int embedding_count;
    };

private:
    cv::Ptr<cv::FaceDetectorYN> detector_;
    cv::Ptr<cv::FaceRecognizerSF> recognizer_;
    std::shared_ptr<FaceDatabase> database_;
    
    std::string detector_model_path_;
    std::string recognizer_model_path_;
    
    float detect_threshold_ = 0.6f;
    float nms_threshold_ = 0.3f;
    int input_width_ = 320;
    int input_height_ = 320;

    mutable std::vector<float> last_embedding_;  // Store last extracted embedding

public:
    FaceService() = default;

    /**
     * @brief Initialize the face service with models
     */
    bool initialize(
        const std::string& detector_model,
        const std::string& recognizer_model,
        std::shared_ptr<FaceDatabase> database
    ) {
        detector_model_path_ = detector_model;
        recognizer_model_path_ = recognizer_model;
        database_ = database;

        // Create face detector
        try {
            detector_ = cv::FaceDetectorYN::create(
                detector_model,
                "",
                cv::Size(input_width_, input_height_),
                detect_threshold_,
                nms_threshold_,
                5000,
                cv::dnn::DNN_BACKEND_OPENCV,
                cv::dnn::DNN_TARGET_CPU
            );
        } catch (const cv::Exception& e) {
            std::cerr << "Failed to create face detector: " << e.what() << std::endl;
            return false;
        }

        if (!detector_) {
            std::cerr << "Face detector is null" << std::endl;
            return false;
        }

        // Create face recognizer
        try {
            recognizer_ = cv::FaceRecognizerSF::create(recognizer_model, "");
        } catch (const cv::Exception& e) {
            std::cerr << "Failed to create face recognizer: " << e.what() << std::endl;
            return false;
        }

        if (!recognizer_) {
            std::cerr << "Face recognizer is null" << std::endl;
            return false;
        }

        return true;
    }

    /**
     * @brief Detect faces in an image
     */
    std::vector<FaceInfo> detect_faces(const cv::Mat& image) {
        std::vector<FaceInfo> faces;
        
        if (image.empty() || !detector_) {
            return faces;
        }

        try {
            detector_->setInputSize(image.size());
            
            cv::Mat detections;
            detector_->detect(image, detections);

            for (int i = 0; i < detections.rows; i++) {
                FaceInfo face;
                face.box.x = static_cast<int>(detections.at<float>(i, 0));
                face.box.y = static_cast<int>(detections.at<float>(i, 1));
                face.box.width = static_cast<int>(detections.at<float>(i, 2));
                face.box.height = static_cast<int>(detections.at<float>(i, 3));
                face.confidence = detections.at<float>(i, 14);

                // Extract 5 landmarks (10 values: x1,y1,x2,y2,...)
                for (int j = 4; j < 14; j++) {
                    face.landmarks.push_back(detections.at<float>(i, j));
                }

                faces.push_back(face);
            }
        } catch (const cv::Exception& e) {
            std::cerr << "Face detection error: " << e.what() << std::endl;
        }

        return faces;
    }

    /**
     * @brief Extract face embedding
     */
    std::vector<float> extract_embedding(const cv::Mat& image, const cv::Mat& face_detection) {
        std::vector<float> embedding;
        
        if (image.empty() || !recognizer_ || face_detection.empty()) {
            return embedding;
        }

        try {
            cv::Mat aligned_face;
            recognizer_->alignCrop(image, face_detection, aligned_face);

            cv::Mat feature;
            recognizer_->feature(aligned_face, feature);

            if (!feature.empty()) {
                embedding.assign(feature.begin<float>(), feature.end<float>());
            }
        } catch (const cv::Exception& e) {
            std::cerr << "Embedding extraction error: " << e.what() << std::endl;
        }

        return embedding;
    }

    /**
     * @brief Get aligned face image
     */
    cv::Mat get_aligned_face(const cv::Mat& image, const cv::Mat& face_detection) {
        cv::Mat aligned;
        if (image.empty() || !recognizer_ || face_detection.empty()) {
            return aligned;
        }

        try {
            recognizer_->alignCrop(image, face_detection, aligned);
        } catch (const cv::Exception& e) {
            std::cerr << "Face alignment error: " << e.what() << std::endl;
        }

        return aligned;
    }

    /**
     * @brief Register a face with augmentation
     */
    RegisterResult register_face(const cv::Mat& image, const std::string& name) {
        RegisterResult result{false, "", name, 0};
        
        if (image.empty()) {
            result.message = "Empty image";
            return result;
        }

        // Detect faces
        try {
            detector_->setInputSize(image.size());
        } catch (const cv::Exception& e) {
            result.message = "Failed to set input size: " + std::string(e.what());
            return result;
        }

        cv::Mat detections;
        try {
            detector_->detect(image, detections);
        } catch (const cv::Exception& e) {
            result.message = "Detection failed: " + std::string(e.what());
            return result;
        }

        if (detections.rows == 0) {
            result.message = "No face detected in image";
            return result;
        }

        // Use first detected face
        cv::Mat face_det = detections.row(0);
        
        // Get aligned face
        cv::Mat aligned_face;
        try {
            recognizer_->alignCrop(image, face_det, aligned_face);
        } catch (const cv::Exception& e) {
            result.message = "Face alignment failed: " + std::string(e.what());
            return result;
        }

        if (aligned_face.empty()) {
            result.message = "Failed to align face";
            return result;
        }

        // Extract embeddings with augmentation
        std::vector<std::vector<float>> all_embeddings;

        // Helper lambda to extract embedding
        auto extract = [this](const cv::Mat& face) -> std::vector<float> {
            cv::Mat input = face;
            if (input.rows != 112 || input.cols != 112) {
                cv::resize(face, input, cv::Size(112, 112));
            }
            
            cv::Mat feature;
            try {
                recognizer_->feature(input, feature);
                if (!feature.empty()) {
                    return std::vector<float>(feature.begin<float>(), feature.end<float>());
                }
            } catch (...) {}
            return {};
        };

        // 1. Original face
        auto emb = extract(aligned_face);
        if (!emb.empty()) all_embeddings.push_back(emb);

        // 2. Horizontal flip
        cv::Mat flipped;
        cv::flip(aligned_face, flipped, 1);
        emb = extract(flipped);
        if (!emb.empty()) all_embeddings.push_back(emb);

        // 3. Brightness variations
        cv::Mat bright, dark;
        aligned_face.convertTo(bright, -1, 1.0, 15);
        aligned_face.convertTo(dark, -1, 1.0, -15);
        
        emb = extract(bright);
        if (!emb.empty()) all_embeddings.push_back(emb);
        
        emb = extract(dark);
        if (!emb.empty()) all_embeddings.push_back(emb);

        // 4. Contrast variation
        cv::Mat contrast;
        aligned_face.convertTo(contrast, -1, 1.1, 0);
        emb = extract(contrast);
        if (!emb.empty()) all_embeddings.push_back(emb);

        if (all_embeddings.empty()) {
            result.message = "Failed to extract any embeddings";
            return result;
        }

        // Average all embeddings
        size_t emb_dim = all_embeddings[0].size();
        std::vector<float> avg_embedding(emb_dim, 0.0f);

        for (const auto& e : all_embeddings) {
            for (size_t i = 0; i < emb_dim; i++) {
                avg_embedding[i] += e[i];
            }
        }

        float count = static_cast<float>(all_embeddings.size());
        for (size_t i = 0; i < emb_dim; i++) {
            avg_embedding[i] /= count;
        }

        // L2 normalize
        float norm = 0.0f;
        for (float v : avg_embedding) norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 1e-6f) {
            for (float& v : avg_embedding) v /= norm;
        }

        // Add to database
        database_->add_embedding(name, avg_embedding);
        database_->save();

        // Store last embedding for retrieval
        last_embedding_ = avg_embedding;

        result.success = true;
        result.message = "Registered " + name + " with " + 
                         std::to_string(all_embeddings.size()) + " augmented variants";
        result.embedding_count = static_cast<int>(all_embeddings.size());
        
        return result;
    }

    /**
     * @brief Get the last extracted/registered embedding
     */
    std::vector<float> get_last_embedding() const {
        return last_embedding_;
    }

    /**
     * @brief Extract face embedding from image without saving
     */
    std::vector<float> extract_face_embedding(const cv::Mat& image) {
        std::vector<float> result;
        
        if (image.empty() || !detector_ || !recognizer_) {
            return result;
        }

        try {
            detector_->setInputSize(image.size());
            cv::Mat detections;
            detector_->detect(image, detections);

            if (detections.rows == 0) {
                return result;
            }

            cv::Mat face_det = detections.row(0);
            cv::Mat aligned_face;
            recognizer_->alignCrop(image, face_det, aligned_face);

            if (aligned_face.empty()) {
                return result;
            }

            // Extract single embedding (no augmentation for simplicity)
            cv::Mat input = aligned_face;
            if (input.rows != 112 || input.cols != 112) {
                cv::resize(aligned_face, input, cv::Size(112, 112));
            }
            
            cv::Mat feature;
            recognizer_->feature(input, feature);
            
            if (!feature.empty()) {
                result.assign(feature.begin<float>(), feature.end<float>());
                last_embedding_ = result;
            }
        } catch (const cv::Exception& e) {
            std::cerr << "Embedding extraction error: " << e.what() << std::endl;
        }

        return result;
    }

    /**
     * @brief Recognize faces in an image
     */
    std::vector<FaceInfo> recognize_faces(const cv::Mat& image) {
        std::vector<FaceInfo> results;
        
        if (image.empty() || !detector_ || !recognizer_) {
            return results;
        }

        try {
            detector_->setInputSize(image.size());
            
            cv::Mat detections;
            detector_->detect(image, detections);

            for (int i = 0; i < detections.rows; i++) {
                FaceInfo face;
                face.box.x = static_cast<int>(detections.at<float>(i, 0));
                face.box.y = static_cast<int>(detections.at<float>(i, 1));
                face.box.width = static_cast<int>(detections.at<float>(i, 2));
                face.box.height = static_cast<int>(detections.at<float>(i, 3));
                face.confidence = detections.at<float>(i, 14);

                // Extract embedding
                cv::Mat face_det = detections.row(i);
                cv::Mat aligned_face;
                recognizer_->alignCrop(image, face_det, aligned_face);

                cv::Mat feature;
                recognizer_->feature(aligned_face, feature);

                if (!feature.empty()) {
                    face.embedding.assign(feature.begin<float>(), feature.end<float>());
                    
                    // Identify
                    auto match = database_->identify(face.embedding);
                    face.name = match.name;
                    face.similarity = match.similarity;
                }

                results.push_back(face);
            }
        } catch (const cv::Exception& e) {
            std::cerr << "Recognition error: " << e.what() << std::endl;
        }

        return results;
    }

    bool is_initialized() const {
        return detector_ && recognizer_ && database_;
    }
};
