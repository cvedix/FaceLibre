/**
 * @file main.cpp
 * @brief Face Recognition REST API Server using Drogon
 * 
 * REST API endpoints for face registration and recognition.
 * 
 * Usage:
 *   ./face_api_server [port]
 * 
 * Default port: 8080
 */

#include <iostream>
#include <fstream>
#include <filesystem>
#include <memory>
#include <random>
#include <set>

#include <drogon/drogon.h>
#include <nlohmann/json.hpp>

#include "face_database.hpp"
#include "face_database_mysql.hpp"
#include "face_service.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace drogon;

// Global instances
std::shared_ptr<FaceDatabase> g_database;
std::shared_ptr<FaceDatabaseMySQL> g_database_mysql;
std::shared_ptr<FaceService> g_face_service;
bool g_use_mysql = false;

// ========================================
// DATABASE HELPER FUNCTIONS
// ========================================

/**
 * @brief Add face to database (file or MySQL) with device info
 */
inline std::string db_add_face(const std::string& name, const std::vector<float>& embedding, 
                               const std::string& base64 = "",
                               const std::string& mac_address = "",
                               const std::string& machine_id = "") {
    if (g_use_mysql && g_database_mysql) {
        return g_database_mysql->add_face(name, embedding, base64, mac_address, machine_id);
    }
    return g_database->add_face(name, embedding, base64);
}

/**
 * @brief Get all faces from database
 */
inline std::vector<FaceDatabase::FaceEntry> db_get_all_faces() {
    if (g_use_mysql && g_database_mysql) {
        // Convert MySQL entries to file database type
        auto mysql_faces = g_database_mysql->get_all_faces();
        std::vector<FaceDatabase::FaceEntry> result;
        for (const auto& f : mysql_faces) {
            FaceDatabase::FaceEntry entry;
            entry.image_id = f.image_id;
            entry.name = f.name;
            entry.base64_image = f.base64_image;
            entry.embedding = f.embedding;
            result.push_back(entry);
        }
        return result;
    }
    return g_database->get_all_faces();
}

/**
 * @brief Get all persons from database
 */
inline std::vector<FaceDatabase::PersonInfo> db_get_all_persons() {
    if (g_use_mysql && g_database_mysql) {
        // Convert MySQL entries to file database type
        auto mysql_persons = g_database_mysql->get_all_persons();
        std::vector<FaceDatabase::PersonInfo> result;
        for (const auto& p : mysql_persons) {
            result.push_back({p.name, p.embedding_count});
        }
        return result;
    }
    return g_database->get_all_persons();
}

/**
 * @brief Delete person from database
 */
inline bool db_delete_person(const std::string& name) {
    if (g_use_mysql && g_database_mysql) {
        return g_database_mysql->delete_person(name);
    }
    return g_database->delete_person(name);
}

/**
 * @brief Rename person in database
 */
inline bool db_rename_person(const std::string& old_name, const std::string& new_name) {
    if (g_use_mysql && g_database_mysql) {
        return g_database_mysql->rename_person(old_name, new_name);
    }
    return g_database->rename_person(old_name, new_name);
}

/**
 * @brief Save database (no-op for MySQL)
 */
inline bool db_save() {
    if (g_use_mysql && g_database_mysql) {
        return g_database_mysql->save();
    }
    return g_database->save();
}

/**
 * @brief Get person count
 */
inline size_t db_person_count() {
    if (g_use_mysql && g_database_mysql) {
        return g_database_mysql->person_count();
    }
    return g_database->person_count();
}

/**
 * @brief Get embedding count
 */
inline size_t db_embedding_count() {
    if (g_use_mysql && g_database_mysql) {
        return g_database_mysql->embedding_count();
    }
    return g_database->embedding_count();
}

// ========================================
// DEVICE INFO HELPERS
// ========================================

/**
 * @brief Execute shell command and return output
 */
std::string get_command_output(const char* cmd) {
    std::array<char, 256> buffer;
    std::string result;
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return "";
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }
    pclose(pipe);
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
        result.pop_back();
    }
    return result;
}

/**
 * @brief Get MAC address (cached)
 */
std::string get_mac_address() {
    static std::string cached_mac;
    if (cached_mac.empty()) {
        cached_mac = get_command_output("cat /sys/class/net/$(ip route show default | awk '/default/ {print $5}' | head -1)/address 2>/dev/null || echo 'unknown'");
    }
    return cached_mac;
}

/**
 * @brief Get machine ID (cached)
 */
std::string get_machine_id() {
    static std::string cached_id;
    if (cached_id.empty()) {
        std::ifstream mid_file("/etc/machine-id");
        if (mid_file.is_open()) {
            std::getline(mid_file, cached_id);
            mid_file.close();
        }
    }
    return cached_id;
}

/**
 * @brief Log request to database (synchronous to avoid memory issues)
 */
void log_to_db(const std::string& type, const std::string& client_ip,
               const std::string& req_body, const std::string& resp_body, 
               int code, const std::string& notes = "") {
    try {
        std::string mac_addr = get_mac_address();
        std::string machine_id = get_machine_id();
        
        if (g_use_mysql && g_database_mysql) {
            g_database_mysql->log_request(type, client_ip, mac_addr, machine_id, 
                                          req_body, resp_body, code, notes);
        } else {
            // File-based logging
            auto now = std::chrono::system_clock::now();
            auto time_t_now = std::chrono::system_clock::to_time_t(now);
            std::tm tm_now{};
            localtime_r(&time_t_now, &tm_now);
            
            char timestamp[64];
            std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", &tm_now);
            
            Json::Value log_entry;
            log_entry["timestamp"] = timestamp;
            log_entry["request_type"] = type;
            log_entry["client_ip"] = client_ip;
            log_entry["mac_address"] = mac_addr;
            log_entry["machine_id"] = machine_id;
            log_entry["response_code"] = code;
            log_entry["notes"] = notes;
            log_entry["request_body"] = req_body;
            log_entry["response_body"] = resp_body;
            
            std::string log_path = "./data/api_log.txt";
            std::ofstream log_file(log_path, std::ios::app);
            if (log_file.is_open()) {
                Json::StreamWriterBuilder builder;
                builder["indentation"] = "";
                log_file << Json::writeString(builder, log_entry) << "\n";
                log_file.close();
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Logging error: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown logging error" << std::endl;
    }
}

/**
 * @brief JSON error response helper
 */
Json::Value error_json(const std::string& message) {
    Json::Value json;
    json["success"] = false;
    json["error"] = message;
    return json;
}

/**
 * @brief Convert nlohmann::json to Json::Value (jsoncpp)
 */
Json::Value to_jsoncpp(const json& j) {
    Json::Value result;
    Json::CharReaderBuilder builder;
    std::string errors;
    std::istringstream ss(j.dump());
    Json::parseFromStream(builder, ss, &result, &errors);
    return result;
}

/**
 * @brief Convert Json::Value (jsoncpp) to string
 */
std::string jsoncpp_to_string(const Json::Value& json) {
    Json::StreamWriterBuilder builder;
    builder["indentation"] = ""; // Compact JSON
    return Json::writeString(builder, json);
}


/**
 * @brief Decode base64 string to bytes
 */
std::vector<unsigned char> decode_base64(const std::string& encoded) {
    static const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    
    // Handle data URL format: data:image/jpeg;base64,xxxxx
    std::string data = encoded;
    size_t comma_pos = data.find(',');
    if (comma_pos != std::string::npos) {
        data = data.substr(comma_pos + 1);
    }
    
    std::vector<unsigned char> decoded;
    int val = 0, valb = -8;
    
    for (unsigned char c : data) {
        if (c == '=') break;
        if (c == '\n' || c == '\r' || c == ' ') continue;  // Skip whitespace
        size_t pos = base64_chars.find(c);
        if (pos == std::string::npos) continue;
        val = (val << 6) + static_cast<int>(pos);
        valb += 6;
        if (valb >= 0) {
            decoded.push_back(static_cast<unsigned char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return decoded;
}

// ========================================
// API CONTROLLERS
// ========================================

/**
 * @brief Health check controller
 */
class HealthController : public HttpController<HealthController> {
public:
    METHOD_LIST_BEGIN
    ADD_METHOD_TO(HealthController::health, "/api/health", Get);
    METHOD_LIST_END

    void health(const HttpRequestPtr& req, 
                std::function<void(const HttpResponsePtr&)>&& callback) {
        json response = {
            {"success", true},
            {"status", "healthy"},
            {"service", "FaceLibre API"},
            {"version", "1.0.0"},
            {"framework", "Drogon"},
            {"database", {
                {"persons", db_person_count()},
                {"embeddings", db_embedding_count()}
            }}
        };
        
        auto resp = HttpResponse::newHttpJsonResponse(to_jsoncpp(response));
        callback(resp);
    }
};



/**
 * @brief Face operations controller
 */
class FaceController : public HttpController<FaceController> {
public:
    METHOD_LIST_BEGIN
    ADD_METHOD_TO(FaceController::getSystemInfo, "/api/v1/recognition/info", Get);
    ADD_METHOD_TO(FaceController::searchFaces, "/api/v1/recognition/search", Post);
    ADD_METHOD_TO(FaceController::registerFaceFaceLibre, "/api/v1/recognition/faces", Post);
    ADD_METHOD_TO(FaceController::listFaces, "/api/v1/recognition/faces", Get);
    ADD_METHOD_TO(FaceController::recognizeFaceLibre, "/api/v1/recognition/recognize", Post);
    ADD_METHOD_TO(FaceController::deleteFacesFaceLibre, "/api/v1/recognition/faces", Delete);
    ADD_METHOD_TO(FaceController::renameSubjectFaceLibre, "/api/v1/recognition/subjects/{subject_name}", Put);
    METHOD_LIST_END

    /**
     * @brief Helper to execute command and get output
     */
    static std::string exec_command(const char* cmd) {
        std::array<char, 256> buffer;
        std::string result;
        FILE* pipe = popen(cmd, "r");
        if (!pipe) return "";
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            result += buffer.data();
        }
        pclose(pipe);
        // Trim trailing newline
        while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
            result.pop_back();
        }
        return result;
    }

    /**
     * @brief Get system information
     * 
     * GET /api/v1/recognition/info
     */
    void getSystemInfo(const HttpRequestPtr& req,
                       std::function<void(const HttpResponsePtr&)>&& callback) {
        Json::Value response;
        response["success"] = true;
        
        // Get hostname
        response["hostname"] = exec_command("hostname");
        
        // Get machine ID (unique per installation)
        std::string machine_id;
        std::ifstream mid_file("/etc/machine-id");
        if (mid_file.is_open()) {
            std::getline(mid_file, machine_id);
            mid_file.close();
        }
        response["machine_id"] = machine_id;
        
        // Get MAC address of first network interface
        std::string mac_address = exec_command("cat /sys/class/net/$(ip route show default | awk '/default/ {print $5}' | head -1)/address 2>/dev/null || echo 'unknown'");
        response["mac_address"] = mac_address;
        
        // Get IP addresses
        std::string ip_addresses = exec_command("hostname -I 2>/dev/null | tr ' ' ','");
        if (!ip_addresses.empty() && ip_addresses.back() == ',') {
            ip_addresses.pop_back();
        }
        response["ip_addresses"] = ip_addresses;
        
        // Get CPU serial (if available)
        std::string cpu_serial = exec_command("cat /proc/cpuinfo | grep Serial | awk '{print $3}' 2>/dev/null");
        if (!cpu_serial.empty()) {
            response["cpu_serial"] = cpu_serial;
        }
        
        // Get OS info
        response["os"] = exec_command("uname -s");
        response["os_version"] = exec_command("uname -r");
        response["architecture"] = exec_command("uname -m");
        
        // Service info
        response["service"] = "FaceLibre API";
        response["version"] = "1.0.0";
        
        // Database info
        Json::Value db_info;
        db_info["type"] = g_use_mysql ? "mysql" : "file";
        db_info["persons"] = static_cast<Json::UInt64>(db_person_count());
        db_info["embeddings"] = static_cast<Json::UInt64>(db_embedding_count());
        response["database"] = db_info;
        
        auto resp = HttpResponse::newHttpJsonResponse(response);
        callback(resp);
    }

    /**
     * @brief Search for similar faces in the database
     * 
     * POST /api/v1/recognition/search
     * Query params: limit (default 10), threshold (default 0.3)
     * Body: {"file": "<base64_image>"}
     */
    void searchFaces(const HttpRequestPtr& req,
                     std::function<void(const HttpResponsePtr&)>&& callback) {
        // Parse query parameters
        int limit = 10;
        float threshold = 0.3f;
        
        std::string limit_str = req->getParameter("limit");
        if (!limit_str.empty()) {
            try { limit = std::stoi(limit_str); } catch (...) {}
        }
        std::string threshold_str = req->getParameter("threshold");
        if (!threshold_str.empty()) {
            try { threshold = std::stof(threshold_str); } catch (...) {}
        }

        // Parse JSON body
        auto jsonPtr = req->getJsonObject();
        if (!jsonPtr || !jsonPtr->isMember("file")) {
            auto resp = HttpResponse::newHttpJsonResponse(error_json("Missing 'file' field"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        std::string base64_image = (*jsonPtr)["file"].asString();
        if (base64_image.empty()) {
            auto resp = HttpResponse::newHttpJsonResponse(error_json("Empty 'file' field"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        // Decode base64
        std::vector<unsigned char> imageData = decode_base64(base64_image);
        if (imageData.empty()) {
            auto resp = HttpResponse::newHttpJsonResponse(error_json("Failed to decode base64 image"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        // Decode image
        cv::Mat image = cv::imdecode(imageData, cv::IMREAD_COLOR);
        if (image.empty()) {
            auto resp = HttpResponse::newHttpJsonResponse(error_json("Failed to decode image"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        LOG_INFO << "[FaceLibre] Searching similar faces in image: " 
                 << image.cols << "x" << image.rows 
                 << " (limit=" << limit << ", threshold=" << threshold << ")";

        // Extract face embedding
        std::vector<float> query_embedding = g_face_service->extract_face_embedding(image);
        if (query_embedding.empty()) {
            auto resp = HttpResponse::newHttpJsonResponse(error_json("No face detected in image"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        // Get all faces and calculate similarity
        auto all_faces = db_get_all_faces();
        
        // Store results: similarity, index
        std::vector<std::pair<float, size_t>> scores;
        scores.reserve(all_faces.size());
        
        for (size_t idx = 0; idx < all_faces.size(); idx++) {
            if (query_embedding.size() != all_faces[idx].embedding.size()) continue;
            
            // Calculate cosine similarity
            float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
            for (size_t i = 0; i < query_embedding.size(); i++) {
                dot += query_embedding[i] * all_faces[idx].embedding[i];
                norm_a += query_embedding[i] * query_embedding[i];
                norm_b += all_faces[idx].embedding[i] * all_faces[idx].embedding[i];
            }
            float similarity = dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-6f);
            
            if (similarity >= threshold) {
                scores.push_back({similarity, idx});
            }
        }

        // Sort by similarity descending
        std::sort(scores.begin(), scores.end(), 
            [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) { 
                return a.first > b.first; 
            });

        // Limit results
        if (limit > 0 && static_cast<int>(scores.size()) > limit) {
            scores.resize(limit);
        }

        LOG_INFO << "[FaceLibre] Search found " << scores.size() << " similar faces";

        // Build response
        Json::Value response;
        response["success"] = true;
        response["total_found"] = static_cast<Json::UInt64>(scores.size());
        
        Json::Value matches(Json::arrayValue);
        for (size_t i = 0; i < scores.size(); i++) {
            size_t idx = scores[i].second;
            Json::Value match;
            match["image_id"] = all_faces[idx].image_id;
            match["subject"] = all_faces[idx].name;
            match["similarity"] = scores[i].first;
            match["base64"] = all_faces[idx].base64_image;
            matches.append(match);
        }
        response["matches"] = matches;

        auto resp = HttpResponse::newHttpJsonResponse(response);
        callback(resp);
    }

    /**
     * @brief Generate simple UUID-like string
     */
    static std::string generate_uuid() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dis(0, 15);
        static const char* hex = "0123456789abcdef";
        
        std::string uuid = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx";
        for (char& c : uuid) {
            if (c == 'x') {
                c = hex[dis(gen)];
            } else if (c == 'y') {
                c = hex[(dis(gen) & 0x3) | 0x8];
            }
        }
        return uuid;
    }

    /**
     * @brief List all registered faces (FaceLibre-compatible)
     * 
     * GET /api/v1/recognition/faces
     */
    void listFaces(const HttpRequestPtr& req,
                   std::function<void(const HttpResponsePtr&)>&& callback) {
        // Log API key if present (optional)
        std::string apiKey = req->getHeader("x-api-key");
        if (!apiKey.empty()) {
            LOG_DEBUG << "API key provided for list: " << apiKey.substr(0, 8) << "...";
        }

        auto faces = db_get_all_faces();
        
        json faces_array = json::array();
        for (const auto& f : faces) {
            faces_array.push_back({
                {"image_id", f.image_id},
                {"subject", f.name},
                {"base64", f.base64_image}
            });
        }

        json response = {
            {"faces", faces_array},
            {"total_elements", faces_array.size()}
        };

        auto resp = HttpResponse::newHttpJsonResponse(to_jsoncpp(response));
        callback(resp);
    }

    /**
     * @brief FaceLibre-compatible face deletion
     * 
     * DELETE /api/v1/recognition/faces?subject=<name>
     * If subject is provided, delete that subject only
     * If subject is empty, delete ALL faces
     */
    void deleteFacesFaceLibre(const HttpRequestPtr& req,
                                std::function<void(const HttpResponsePtr&)>&& callback) {
        // Log API key if present (optional)
        std::string apiKey = req->getHeader("x-api-key");
        if (!apiKey.empty()) {
            LOG_DEBUG << "[FaceLibre] API key provided for delete: " << apiKey.substr(0, 8) << "...";
        }

        // Get subject from query parameter
        std::string subject = req->getParameter("subject");
        
        // Trim whitespace
        subject.erase(0, subject.find_first_not_of(" \t\n\r"));
        if (!subject.empty()) {
            subject.erase(subject.find_last_not_of(" \t\n\r") + 1);
        }

        if (subject.empty()) {
            // Delete ALL faces
            auto faces = db_get_all_faces();
            std::set<std::string> deleted_subjects;
            
            for (const auto& f : faces) {
                if (deleted_subjects.find(f.name) == deleted_subjects.end()) {
                    db_delete_person(f.name);
                    deleted_subjects.insert(f.name);
                }
            }
            db_save();

            json response = {
                {"deleted", deleted_subjects.size()},
                {"message", "All faces deleted"}
            };

            auto resp = HttpResponse::newHttpJsonResponse(to_jsoncpp(response));
            callback(resp);
            LOG_INFO << "[FaceLibre] Deleted all faces (" << deleted_subjects.size() << " subjects)";
        } else {
            // Delete specific subject
            if (db_delete_person(subject)) {
                db_save();
                
                json response = {
                    {"subject", subject},
                    {"deleted", true}
                };

                auto resp = HttpResponse::newHttpJsonResponse(to_jsoncpp(response));
                callback(resp);
                LOG_INFO << "[FaceLibre] Deleted subject: " << subject;
            } else {
                auto resp = HttpResponse::newHttpJsonResponse(
                    error_json("Subject not found: " + subject));
                resp->setStatusCode(k404NotFound);
                callback(resp);
                LOG_WARN << "[FaceLibre] Subject not found for deletion: " << subject;
            }
        }
    }

    /**
     * @brief FaceLibre-compatible subject rename
     * 
     * PUT /api/v1/recognition/subjects/{subject_name}
     * Body: {"subject": "<new_name>"}
     */
    void renameSubjectFaceLibre(const HttpRequestPtr& req,
                                  std::function<void(const HttpResponsePtr&)>&& callback,
                                  const std::string& subject_name) {
        // Log API key if present (optional)
        std::string apiKey = req->getHeader("x-api-key");
        if (!apiKey.empty()) {
            LOG_DEBUG << "[FaceLibre] API key provided for rename: " << apiKey.substr(0, 8) << "...";
        }

        // URL decode the old name
        std::string old_name = utils::urlDecode(subject_name);
        
        // Trim whitespace
        old_name.erase(0, old_name.find_first_not_of(" \t\n\r"));
        if (!old_name.empty()) {
            old_name.erase(old_name.find_last_not_of(" \t\n\r") + 1);
        }

        if (old_name.empty()) {
            auto resp = HttpResponse::newHttpJsonResponse(
                error_json("Missing subject name in URL path"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        // Parse JSON body
        auto jsonPtr = req->getJsonObject();
        if (!jsonPtr) {
            auto resp = HttpResponse::newHttpJsonResponse(
                error_json("Invalid JSON body"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        // Get new name from body
        if (!jsonPtr->isMember("subject")) {
            auto resp = HttpResponse::newHttpJsonResponse(
                error_json("Missing 'subject' field in JSON body"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        std::string new_name = (*jsonPtr)["subject"].asString();
        
        // Trim whitespace
        new_name.erase(0, new_name.find_first_not_of(" \t\n\r"));
        if (!new_name.empty()) {
            new_name.erase(new_name.find_last_not_of(" \t\n\r") + 1);
        }

        if (new_name.empty()) {
            auto resp = HttpResponse::newHttpJsonResponse(
                error_json("Empty 'subject' field in JSON body"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        // Rename in database
        if (db_rename_person(old_name, new_name)) {
            db_save();
            
            json response = {
                {"updated", true},
                {"subject", new_name}
            };

            auto resp = HttpResponse::newHttpJsonResponse(to_jsoncpp(response));
            callback(resp);
            LOG_INFO << "[FaceLibre] Renamed subject: " << old_name << " -> " << new_name;
        } else {
            auto resp = HttpResponse::newHttpJsonResponse(
                error_json("Subject not found: " + old_name));
            resp->setStatusCode(k404NotFound);
            callback(resp);
            LOG_WARN << "[FaceLibre] Subject not found for rename: " << old_name;
        }
    }

    /**
     * @brief FaceLibre-compatible face registration
     * 
     * Endpoint: POST /api/v1/recognition/faces?subject=<name>
     * Headers: x-api-key (optional)
     * Body: {"file": "<base64_image>"}
     */
    void registerFaceFaceLibre(const HttpRequestPtr& req,
                                std::function<void(const HttpResponsePtr&)>&& callback) {
        // Get subject from query parameter
        std::string subject = req->getParameter("subject");
        
        // Trim whitespace
        subject.erase(0, subject.find_first_not_of(" \t\n\r"));
        subject.erase(subject.find_last_not_of(" \t\n\r") + 1);

        if (subject.empty()) {
            auto resp = HttpResponse::newHttpJsonResponse(
                error_json("Missing 'subject' query parameter"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        // Log API key if present (optional, just for compatibility)
        std::string apiKey = req->getHeader("x-api-key");
        if (!apiKey.empty()) {
            LOG_DEBUG << "API key provided: " << apiKey.substr(0, 8) << "...";
        }

        // Parse JSON body
        auto jsonPtr = req->getJsonObject();
        if (!jsonPtr) {
            auto resp = HttpResponse::newHttpJsonResponse(
                error_json("Invalid JSON body"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        // Get base64 file field
        if (!jsonPtr->isMember("file")) {
            auto resp = HttpResponse::newHttpJsonResponse(
                error_json("Missing 'file' field in JSON body"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        std::string base64_image = (*jsonPtr)["file"].asString();
        if (base64_image.empty()) {
            auto resp = HttpResponse::newHttpJsonResponse(
                error_json("Empty 'file' field"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        // Decode base64
        std::vector<unsigned char> imageData = decode_base64(base64_image);
        if (imageData.empty()) {
            auto resp = HttpResponse::newHttpJsonResponse(
                error_json("Failed to decode base64 image"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        // Decode image
        cv::Mat image = cv::imdecode(imageData, cv::IMREAD_COLOR);
        if (image.empty()) {
            auto resp = HttpResponse::newHttpJsonResponse(
                error_json("Failed to decode image from base64 data"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        LOG_INFO << "[FaceLibre] Registering subject: " << subject 
                 << " (image: " << image.cols << "x" << image.rows << ")";

        // Extract face embedding directly
        auto embedding = g_face_service->extract_face_embedding(image);

        if (embedding.empty()) {
            auto resp = HttpResponse::newHttpJsonResponse(
                error_json("No face detected or failed to extract embedding"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            LOG_WARN << "[FaceLibre] Registration failed: no face detected";
            
            // Log failure
            std::string client_ip = req->peerAddr().toIp();
            log_to_db("register", client_ip, jsonPtr ? jsoncpp_to_string(*jsonPtr) : "", 
                      "{\"error\": \"No face detected\"}", 400, subject);
            return;
        }

        // Save to database with base64 image and device info
        std::string mac_addr = get_mac_address();
        std::string machine_id = get_machine_id();
        std::string image_id = db_add_face(subject, embedding, base64_image, mac_addr, machine_id);
        db_save();

        // FaceLibre-style response
        json response = {
            {"image_id", image_id},
            {"subject", subject}
        };
        auto resp = HttpResponse::newHttpJsonResponse(to_jsoncpp(response));
        callback(resp);
        LOG_INFO << "[FaceLibre] Registered: " << subject << " with image_id=" << image_id;

        // Log success
        std::string client_ip = req->peerAddr().toIp();
        log_to_db("register", client_ip, jsonPtr ? jsoncpp_to_string(*jsonPtr) : "", response.dump(), 200, subject);
    }

    void registerFace(const HttpRequestPtr& req,
                      std::function<void(const HttpResponsePtr&)>&& callback) {
        // Method removed
        auto resp = HttpResponse::newHttpJsonResponse(error_json("Endpoint deprecated"));
        resp->setStatusCode(k404NotFound);
        callback(resp);
    }

    void recognize(const HttpRequestPtr& req,
                   std::function<void(const HttpResponsePtr&)>&& callback) {
        // Method removed
        auto resp = HttpResponse::newHttpJsonResponse(error_json("Endpoint deprecated"));
        resp->setStatusCode(k404NotFound);
        callback(resp);
    }

    /**
     * @brief FaceLibre-compatible face recognition
     * 
     * Endpoint: POST /api/v1/recognition/recognize
     * Query params: limit, det_prob_threshold, prediction_count, face_plugins, status
     * Headers: x-api-key (optional)
     * Body: {"file": "<base64_image>"}
     */
    void recognizeFaceLibre(const HttpRequestPtr& req,
                              std::function<void(const HttpResponsePtr&)>&& callback) {
        // Log API key if present (optional)
        std::string apiKey = req->getHeader("x-api-key");
        if (!apiKey.empty()) {
            LOG_DEBUG << "[FaceLibre] API key provided: " << apiKey.substr(0, 8) << "...";
        }

        // Parse query parameters
        int limit = 0;
        double det_prob_threshold = 0.5;  // Lowered to detect faces with sunglasses/masks
        int prediction_count = 1;
        std::string face_plugins = req->getParameter("face_plugins");
        bool return_status = (req->getParameter("status") == "true");

        // Parse optional parameters
        std::string limit_str = req->getParameter("limit");
        if (!limit_str.empty()) {
            try { limit = std::stoi(limit_str); } catch (...) {}
        }
        std::string det_prob_str = req->getParameter("det_prob_threshold");
        if (!det_prob_str.empty()) {
            try { det_prob_threshold = std::stod(det_prob_str); } catch (...) {}
        }
        std::string pred_count_str = req->getParameter("prediction_count");
        if (!pred_count_str.empty()) {
            try { prediction_count = std::stoi(pred_count_str); } catch (...) {}
        }

        // Parse JSON body
        auto jsonPtr = req->getJsonObject();
        if (!jsonPtr) {
            auto resp = HttpResponse::newHttpJsonResponse(
                error_json("Invalid JSON body"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        // Get base64 file field
        if (!jsonPtr->isMember("file")) {
            auto resp = HttpResponse::newHttpJsonResponse(
                error_json("Missing 'file' field in JSON body"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        std::string base64_image = (*jsonPtr)["file"].asString();
        if (base64_image.empty()) {
            auto resp = HttpResponse::newHttpJsonResponse(
                error_json("Empty 'file' field"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        // Decode base64
        std::vector<unsigned char> imageData = decode_base64(base64_image);
        if (imageData.empty()) {
            auto resp = HttpResponse::newHttpJsonResponse(
                error_json("Failed to decode base64 image"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        // Decode image
        cv::Mat image = cv::imdecode(imageData, cv::IMREAD_COLOR);
        if (image.empty()) {
            auto resp = HttpResponse::newHttpJsonResponse(
                error_json("Failed to decode image from base64 data"));
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        LOG_INFO << "[FaceLibre] Recognizing faces in image: " 
                 << image.cols << "x" << image.rows 
                 << " (det_prob_threshold=" << det_prob_threshold 
                 << ", prediction_count=" << prediction_count << ")";

        // Detect faces and extract embeddings
        auto faces = g_face_service->recognize_faces(image);
        
        LOG_INFO << "[FaceLibre] Detected " << faces.size() << " faces";

        // Build FaceLibre-style response
        json result_array = json::array();
        
        for (auto& face : faces) {
            LOG_INFO << "[FaceLibre] Face: confidence=" << face.confidence 
                     << ", embedding_size=" << face.embedding.size()
                     << ", name=" << face.name 
                     << ", similarity=" << face.similarity;
            
            // Skip faces below detection threshold
            if (face.confidence < det_prob_threshold) {
                LOG_INFO << "[FaceLibre] Skipped: confidence < det_prob_threshold";
                continue;
            }

            // Re-identify using the correct database when MySQL is configured
            if (g_use_mysql && g_database_mysql && !face.embedding.empty()) {
                LOG_INFO << "[FaceLibre] Re-identifying with MySQL database...";
                auto match = g_database_mysql->identify(face.embedding);
                face.name = match.name;
                face.similarity = match.similarity;
                LOG_INFO << "[FaceLibre] MySQL identify result: " << face.name << " (" << face.similarity << ")";
            }

            // Build box object (FaceLibre format: x_min, y_min, x_max, y_max)
            json box = {
                {"probability", face.confidence},
                {"x_min", face.box.x},
                {"y_min", face.box.y},
                {"x_max", face.box.x + face.box.width},
                {"y_max", face.box.y + face.box.height}
            };

            // Build subjects array (limit by prediction_count)
            json subjects = json::array();
            if (face.name != "Unknown" && face.similarity > 0) {
                subjects.push_back({
                    {"subject", face.name},
                    {"similarity", face.similarity}
                });
            }

            // Limit predictions if needed
            if (prediction_count > 0 && subjects.size() > static_cast<size_t>(prediction_count)) {
                subjects = json(subjects.begin(), subjects.begin() + prediction_count);
            }

            json face_result = {
                {"box", box},
                {"subjects", subjects}
            };

            // Add plugins data if requested
            if (!face_plugins.empty()) {
                // Landmarks plugin
                if (face_plugins.find("landmarks") != std::string::npos) {
                    json landmarks = json::array();
                    // Return 5-point landmarks if available
                    if (face.landmarks.size() >= 10) {
                        for (size_t i = 0; i < face.landmarks.size(); i += 2) {
                            landmarks.push_back(json::array({face.landmarks[i], face.landmarks[i+1]}));
                        }
                    }
                    face_result["landmarks"] = landmarks;
                }

                // Age plugin (placeholder - would need separate model)
                if (face_plugins.find("age") != std::string::npos) {
                    face_result["age"] = {
                        {"probability", 0.0},
                        {"high", 0},
                        {"low", 0}
                    };
                }

                // Gender plugin (placeholder - would need separate model)
                if (face_plugins.find("gender") != std::string::npos) {
                    face_result["gender"] = {
                        {"probability", 0.0},
                        {"value", ""}
                    };
                }

                // Mask plugin (placeholder - would need separate model)
                if (face_plugins.find("mask") != std::string::npos) {
                    face_result["mask"] = {
                        {"probability", 0.0},
                        {"value", ""}
                    };
                }

                // Pose plugin (placeholder - would need separate model)
                if (face_plugins.find("pose") != std::string::npos) {
                    face_result["pose"] = {
                        {"pitch", 0.0},
                        {"roll", 0.0},
                        {"yaw", 0.0}
                    };
                }

                // Calculator plugin (embedding) - placeholder
                if (face_plugins.find("calculator") != std::string::npos) {
                    face_result["embedding"] = json::array();
                }
            }

            result_array.push_back(face_result);

            // Apply limit if set
            if (limit > 0 && result_array.size() >= static_cast<size_t>(limit)) {
                break;
            }
        }

        json response = {
            {"result", result_array}
        };

        // Add plugins_versions if status requested
        if (return_status) {
            response["plugins_versions"] = {
                {"calculator", "1.0.0"},
                {"age", "1.0.0"},
                {"gender", "1.0.0"},
                {"mask", "1.0.0"},
                {"landmarks", "1.0.0"},
                {"pose", "1.0.0"}
            };
        }

        auto resp = HttpResponse::newHttpJsonResponse(to_jsoncpp(response));
        callback(resp);
        
        LOG_INFO << "[FaceLibre] Recognized " << result_array.size() << " faces";

        // Log result with client IP
        std::string client_ip = req->peerAddr().toIp();
        log_to_db("recognize", client_ip, jsonPtr ? jsoncpp_to_string(*jsonPtr) : "", response.dump(), 200, 
                  "Found " + std::to_string(result_array.size()) + " faces");
    }


};

/**
 * @brief Root controller for documentation
 */
class RootController : public HttpController<RootController> {
public:
    METHOD_LIST_BEGIN
    ADD_METHOD_TO(RootController::index, "/", Get);
    METHOD_LIST_END

    void index(const HttpRequestPtr& req,
               std::function<void(const HttpResponsePtr&)>&& callback) {
        std::string html = R"(
<!DOCTYPE html>
<html>
<head>
    <title>FaceLibre API</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        .badge { background: #28a745; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.8em; }
        .endpoint { background: white; padding: 20px; margin: 15px 0; border-radius: 8px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .method { display: inline-block; padding: 4px 12px; border-radius: 4px; 
                  font-weight: bold; color: white; margin-right: 10px; }
        .get { background: #28a745; }
        .post { background: #007bff; }
        .delete { background: #dc3545; }
        .path { font-family: monospace; font-size: 1.1em; }
        code { background: #e9ecef; padding: 2px 6px; border-radius: 3px; }
        pre { background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 5px; 
              overflow-x: auto; }
    </style>
</head>
<body>
    <h1>🎭 FaceLibre REST API <span class="badge">Drogon</span></h1>
    <p>High-performance face recognition API powered by OpenCV and Drogon framework.</p>
    
    <div class="endpoint">
        <span class="method get">GET</span>
        <span class="path">/api/health</span>
        <p>Health check and service status</p>
    </div>
    
    <h2>FaceLibre Compatible API</h2>

    <div class="endpoint">
        <span class="method get">GET</span>
        <span class="path">/api/v1/recognition/faces</span>
        <p>List all registered faces</p>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span>
        <span class="path">/api/v1/recognition/faces</span>
        <p>Register a face. Query param <code>subject</code>, Body: <code>{"file": "base64"}</code></p>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span>
        <span class="path">/api/v1/recognition/recognize</span>
        <p>Recognize faces. Body: <code>{"file": "base64"}</code></p>
    </div>

    <div class="endpoint">
        <span class="method delete">DELETE</span>
        <span class="path">/api/v1/recognition/faces</span>
        <p>Delete faces. Query param <code>subject</code> (optional)</p>
    </div>

    <div class="endpoint">
        <span class="method put">PUT</span>
        <span class="path">/api/v1/recognition/subjects/:name</span>
        <p>Rename subject. Body: <code>{"subject": "new_name"}</code></p>
    </div>
</body>
</html>
)";
        auto resp = HttpResponse::newHttpResponse();
        resp->setBody(html);
        resp->setContentTypeCode(CT_TEXT_HTML);
        callback(resp);
    }
};

// ========================================
// MAIN
// ========================================

int main(int argc, char* argv[]) {
    std::cout << R"(
╔═══════════════════════════════════════════════════════╗
║     FaceLibre - Face Recognition REST API (Drogon)    ║
╚═══════════════════════════════════════════════════════╝
)" << std::endl;

    // Default values
    int port = 8080;
    bool use_mysql = false;
    std::string mysql_host = "127.0.0.1";
    int mysql_port = 3306;
    std::string mysql_user = "root";
    std::string mysql_pass = "Admin123456";
    std::string mysql_db = "facelibre";
    std::string mysql_table = "faces";
    std::string mysql_log_table = "tc_face_recognition_log";
    
    std::string models_dir = "./models";
    std::string data_dir = "./data";
    std::string detector_model = models_dir + "/face_detection_yunet_2023mar_int8.onnx";
    std::string recognizer_model = models_dir + "/face_recognition_sface_2021dec.onnx";
    
    float recognition_threshold = 0.4f;
    float recognition_min_gap = 0.08f;

    // Load config.json if exists
    if (fs::exists("config.json")) {
        std::cout << "Loading configuration from config.json...\n";
        try {
            std::ifstream f("config.json");
            json config = json::parse(f);
            
            if (config.contains("server")) {
                auto& server = config["server"];
                if (server.contains("port")) port = server["port"];
            }
            
            if (config.contains("database")) {
                auto& db = config["database"];
                if (db.contains("type") && db["type"] == "mysql") {
                    use_mysql = true;
                    if (db.contains("mysql")) {
                        auto& mysql = db["mysql"];
                        if (mysql.contains("host")) mysql_host = mysql["host"];
                        if (mysql.contains("port")) mysql_port = mysql["port"];
                        if (mysql.contains("user")) mysql_user = mysql["user"];
                        if (mysql.contains("password")) mysql_pass = mysql["password"];
                        if (mysql.contains("database")) mysql_db = mysql["database"];
                        if (mysql.contains("table")) mysql_table = mysql["table"];
                        if (mysql.contains("log_table")) mysql_log_table = mysql["log_table"];
                    }
                }
            }
            
            if (config.contains("models")) {
                auto& models = config["models"];
                if (models.contains("detector")) detector_model = models["detector"];
                if (models.contains("recognizer")) recognizer_model = models["recognizer"];
            }

            if (config.contains("recognition")) {
                auto& rec = config["recognition"];
                if (rec.contains("threshold")) recognition_threshold = rec["threshold"];
                if (rec.contains("min_gap")) recognition_min_gap = rec["min_gap"];
            }
        } catch (const std::exception& e) {
            std::cerr << "Error loading config.json: " << e.what() << "\n";
        }
    }

    // Parse arguments (override config)
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--mysql") {
            use_mysql = true;
        } else if (arg.find("--mysql-host=") == 0) {
            mysql_host = arg.substr(13);
        } else if (arg.find("--mysql-port=") == 0) {
            mysql_port = std::stoi(arg.substr(13));
        } else if (arg.find("--mysql-user=") == 0) {
            mysql_user = arg.substr(13);
        } else if (arg.find("--mysql-pass=") == 0) {
            mysql_pass = arg.substr(13);
        } else if (arg.find("--mysql-db=") == 0) {
            mysql_db = arg.substr(11);
        } else {
            try {
                port = std::stoi(arg);
            } catch (...) {
                std::cerr << "Unknown argument: " << arg << std::endl;
            }
        }
    }

    g_use_mysql = use_mysql;

    g_use_mysql = use_mysql;
    
    // Construct database path from data_dir
    std::string database_path = data_dir + "/face_database.txt";


    // Check models exist
    if (!fs::exists(detector_model)) {
        std::cerr << "❌ Detector model not found: " << detector_model << std::endl;
        std::cerr << "Please download from OpenCV Zoo\n";
        return 1;
    }
    if (!fs::exists(recognizer_model)) {
        std::cerr << "❌ Recognizer model not found: " << recognizer_model << std::endl;
        std::cerr << "Please download from OpenCV Zoo\n";
        return 1;
    }

    // Initialize database based on mode
    if (use_mysql) {
        std::cout << "📦 Using MySQL database: " << mysql_user << "@" << mysql_host << ":" << mysql_port << "/" << mysql_db << " [" << mysql_table << "]\n";
        std::cout << "📝 Log table: " << mysql_log_table << "\n";
        g_database_mysql = std::make_shared<FaceDatabaseMySQL>(mysql_host, mysql_port, mysql_user, mysql_pass, mysql_db, mysql_table, mysql_log_table);
        if (!g_database_mysql->is_connected()) {
            std::cerr << "❌ Failed to connect to MySQL database\n";
            return 1;
        }
        std::cout << "✓ MySQL database connected: " << g_database_mysql->person_count() << " persons, " 
                  << g_database_mysql->embedding_count() << " embeddings\n";
        
        // Create a file database wrapper that won't be used but is needed for FaceService
        fs::create_directories(data_dir);
        g_database = std::make_shared<FaceDatabase>(database_path);
        
        // Apply recognition settings
        g_database_mysql->set_threshold(recognition_threshold);
        g_database_mysql->set_min_gap(recognition_min_gap);
    } else {
        fs::create_directories(data_dir);
        g_database = std::make_shared<FaceDatabase>(database_path);
        std::cout << "📦 Using file database: " << database_path << "\n";
        std::cout << "✓ Database loaded: " << g_database->person_count() << " persons, " 
                  << g_database->embedding_count() << " embeddings\n";
                  
        // Apply recognition settings
        g_database->set_threshold(recognition_threshold);
        g_database->set_min_gap(recognition_min_gap);
    }

    // Initialize face service
    g_face_service = std::make_shared<FaceService>();
    if (!g_face_service->initialize(detector_model, recognizer_model, g_database)) {
        std::cerr << "❌ Failed to initialize face service\n";
        return 1;
    }
    std::cout << "✓ Face service initialized\n";
    std::cout << "  Detector: " << detector_model << "\n";
    std::cout << "  Recognizer: " << recognizer_model << "\n";

    // Configure Drogon
    std::cout << "\n🚀 Starting Drogon server on http://0.0.0.0:" << port << "\n";
    std::cout << "📖 API documentation: http://localhost:" << port << "/\n";
    std::cout << "Press Ctrl+C to stop\n\n";

    drogon::app()
        .setLogLevel(trantor::Logger::kInfo)
        .addListener("0.0.0.0", port)
        .setThreadNum(4)
        .setClientMaxBodySize(50 * 1024 * 1024)  // 50MB max body size for large images
        .setClientMaxMemoryBodySize(50 * 1024 * 1024)  // 50MB in-memory
        .run();

    std::cout << "Server stopped.\n";
    return 0;
}
