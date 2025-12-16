#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <random>
#include <mysql/mysql.h>

/**
 * @brief MySQL-based Face Database for storing and matching face embeddings
 */
class FaceDatabaseMySQL {
public:
    struct IdentifyResult {
        std::string name;
        float similarity;
        bool is_known;
    };

    struct FaceEntry {
        std::string image_id;
        std::string name;
        std::string base64_image;
        std::vector<float> embedding;
    };

    struct PersonInfo {
        std::string name;
        int embedding_count;
    };

    struct LogEntry {
        std::string request_type;
        std::string request_body;
        std::string response_body;
        int response_code;
        std::string notes;
    };

private:
    MYSQL* conn_ = nullptr;
    std::string host_;
    int port_;
    std::string user_;
    std::string password_;
    std::string database_;
    std::string table_name_ = "faces";
    std::string log_table_name_ = "tc_face_recognition_log";
    mutable std::mutex mutex_;
    
    // Recognition parameters
    float threshold_ = 0.4f;
    float min_gap_ = 0.08f;

    /**
     * @brief Generate UUID-like string
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
     * @brief Escape string for MySQL
     */
    std::string escape_string(const std::string& str) {
        if (!conn_) return str;
        std::vector<char> buffer(str.length() * 2 + 1);
        mysql_real_escape_string(conn_, buffer.data(), str.c_str(), str.length());
        return std::string(buffer.data());
    }

    /**
     * @brief Serialize embedding to string
     */
    static std::string serialize_embedding(const std::vector<float>& embedding) {
        std::ostringstream oss;
        for (size_t i = 0; i < embedding.size(); ++i) {
            if (i > 0) oss << ",";
            oss << embedding[i];
        }
        return oss.str();
    }

    /**
     * @brief Parse embedding from string
     */
    static std::vector<float> parse_embedding(const std::string& str) {
        std::vector<float> result;
        std::istringstream iss(str);
        std::string token;
        while (std::getline(iss, token, ',')) {
            try {
                result.push_back(std::stof(token));
            } catch (...) {}
        }
        return result;
    }

public:
    FaceDatabaseMySQL() = default;
    
    FaceDatabaseMySQL(const std::string& host, int port, 
                      const std::string& user, const std::string& password,
                      const std::string& database, const std::string& table_name = "faces",
                      const std::string& log_table_name = "tc_face_recognition_log")
        : host_(host), port_(port), user_(user), password_(password), 
          database_(database), table_name_(table_name), log_table_name_(log_table_name) {
        connect();
    }

    ~FaceDatabaseMySQL() {
        disconnect();
    }

    /**
     * @brief Connect to MySQL server
     */
    bool connect() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (conn_) {
            mysql_close(conn_);
        }

        conn_ = mysql_init(nullptr);
        if (!conn_) {
            return false;
        }

        if (!mysql_real_connect(conn_, host_.c_str(), user_.c_str(), 
                                password_.c_str(), database_.c_str(), 
                                port_, nullptr, 0)) {
            std::cerr << "MySQL connection error: " << mysql_error(conn_) << std::endl;
            mysql_close(conn_);
            conn_ = nullptr;
            return false;
        }

        // Set UTF-8
        mysql_set_character_set(conn_, "utf8mb4");
        
        ensure_log_table();
        
        return true;
    }

    /**
     * @brief Ensure log table exists
     */
    void ensure_log_table() {
        if (!conn_) return;
        
        std::string sql = "CREATE TABLE IF NOT EXISTS " + log_table_name_ + " ("
                          "id INT AUTO_INCREMENT PRIMARY KEY, "
                          "request_type VARCHAR(50), "
                          "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, "
                          "client_ip VARCHAR(45), "
                          "mac_address VARCHAR(20), "
                          "machine_id VARCHAR(64), "
                          "request_body LONGTEXT, "
                          "response_body LONGTEXT, "
                          "response_code INT, "
                          "notes TEXT"
                          ") CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci";
                          
        if (mysql_query(conn_, sql.c_str()) != 0) {
            std::cerr << "MySQL create log table error: " << mysql_error(conn_) << std::endl;
        }

        // Add columns if they don't exist (for existing tables)
        const char* alter_queries[] = {
            "ALTER TABLE %s ADD COLUMN IF NOT EXISTS client_ip VARCHAR(45)",
            "ALTER TABLE %s ADD COLUMN IF NOT EXISTS mac_address VARCHAR(20)",
            "ALTER TABLE %s ADD COLUMN IF NOT EXISTS machine_id VARCHAR(64)"
        };
        for (const char* query_template : alter_queries) {
            char query[512];
            snprintf(query, sizeof(query), query_template, log_table_name_.c_str());
            mysql_query(conn_, query); // Ignore errors (column may already exist)
        }
    }

    /**
     * @brief Log an API request with device info
     */
    bool log_request(const std::string& type, 
                     const std::string& client_ip,
                     const std::string& mac_address,
                     const std::string& machine_id,
                     const std::string& request_body, 
                     const std::string& response_body, 
                     int code, 
                     const std::string& notes = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!conn_) return false;

        std::string sql = "INSERT INTO " + log_table_name_ + 
                          " (request_type, client_ip, mac_address, machine_id, request_body, response_body, response_code, notes) VALUES ('" +
                          escape_string(type) + "', '" +
                          escape_string(client_ip) + "', '" +
                          escape_string(mac_address) + "', '" +
                          escape_string(machine_id) + "', '" +
                          escape_string(request_body) + "', '" +
                          escape_string(response_body) + "', " +
                          std::to_string(code) + ", '" +
                          escape_string(notes) + "')";

        if (mysql_query(conn_, sql.c_str()) != 0) {
            std::cerr << "MySQL log error: " << mysql_error(conn_) << std::endl;
            return false;
        }
        return true;
    }

    /**
     * @brief Disconnect from MySQL server
     */
    void disconnect() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (conn_) {
            mysql_close(conn_);
            conn_ = nullptr;
        }
    }

    /**
     * @brief Check if connected
     */
    bool is_connected() const {
        return conn_ != nullptr && mysql_ping(const_cast<MYSQL*>(conn_)) == 0;
    }

    /**
     * @brief Ensure faces table has mac_address and machine_id columns
     */
    void ensure_face_table_columns() {
        if (!conn_) return;
        
        const char* alter_queries[] = {
            "ALTER TABLE %s ADD COLUMN IF NOT EXISTS mac_address VARCHAR(20)",
            "ALTER TABLE %s ADD COLUMN IF NOT EXISTS machine_id VARCHAR(64)"
        };
        for (const char* query_template : alter_queries) {
            char query[512];
            snprintf(query, sizeof(query), query_template, table_name_.c_str());
            mysql_query(conn_, query); // Ignore errors (column may already exist)
        }
    }

    /**
     * @brief Add a new face entry with device info
     */
    std::string add_face(const std::string& name, const std::vector<float>& embedding,
                         const std::string& base64_image = "",
                         const std::string& mac_address = "",
                         const std::string& machine_id = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!conn_) return "";

        // Ensure columns exist (only first time)
        static bool columns_checked = false;
        if (!columns_checked) {
            ensure_face_table_columns();
            columns_checked = true;
        }

        std::string image_id = generate_uuid();
        std::string emb_str = serialize_embedding(embedding);
        
        std::string sql = "INSERT INTO " + table_name_ + 
                          " (image_id, subject, base64_image, embedding, mac_address, machine_id) VALUES ('" +
                          escape_string(image_id) + "', '" +
                          escape_string(name) + "', '" +
                          escape_string(base64_image) + "', '" +
                          escape_string(emb_str) + "', '" +
                          escape_string(mac_address) + "', '" +
                          escape_string(machine_id) + "')";

        if (mysql_query(conn_, sql.c_str()) != 0) {
            std::cerr << "MySQL insert error: " << mysql_error(conn_) << std::endl;
            return "";
        }

        return image_id;
    }

    /**
     * @brief Add a new embedding for a person (legacy API)
     */
    void add_embedding(const std::string& name, const std::vector<float>& embedding) {
        add_face(name, embedding, "", "", "");
    }

    /**
     * @brief Delete all entries for a person
     */
    bool delete_person(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!conn_) return false;

        std::string sql = "DELETE FROM " + table_name_ + " WHERE subject = '" + escape_string(name) + "'";
        
        if (mysql_query(conn_, sql.c_str()) != 0) {
            std::cerr << "MySQL delete error: " << mysql_error(conn_) << std::endl;
            return false;
        }

        return mysql_affected_rows(conn_) > 0;
    }

    /**
     * @brief Rename a person (update all their entries)
     */
    bool rename_person(const std::string& old_name, const std::string& new_name) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!conn_) return false;

        std::string sql = "UPDATE " + table_name_ + " SET subject = '" + escape_string(new_name) + 
                          "' WHERE subject = '" + escape_string(old_name) + "'";
        
        if (mysql_query(conn_, sql.c_str()) != 0) {
            std::cerr << "MySQL update error: " << mysql_error(conn_) << std::endl;
            return false;
        }

        return mysql_affected_rows(conn_) > 0;
    }

    /**
     * @brief Get all face entries
     */
    std::vector<FaceEntry> get_all_faces() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<FaceEntry> result;
        
        if (!conn_) return result;

        std::string sql = "SELECT image_id, subject, base64_image, embedding FROM " + table_name_;
        
        if (mysql_query(conn_, sql.c_str()) != 0) {
            return result;
        }

        MYSQL_RES* res = mysql_store_result(conn_);
        if (!res) return result;

        MYSQL_ROW row;
        while ((row = mysql_fetch_row(res))) {
            FaceEntry entry;
            entry.image_id = row[0] ? row[0] : "";
            entry.name = row[1] ? row[1] : "";
            entry.base64_image = row[2] ? row[2] : "";
            entry.embedding = parse_embedding(row[3] ? row[3] : "");
            result.push_back(entry);
        }

        mysql_free_result(res);
        return result;
    }

    /**
     * @brief Identify a face by its embedding
     */
    IdentifyResult identify(const std::vector<float>& query_embedding) {
        auto faces = get_all_faces();
        
        if (query_embedding.empty() || faces.empty()) {
            return {"Unknown", 0.0f, false};
        }

        // Calculate max similarity for each person
        std::map<std::string, float> person_max_sim;

        for (const auto& entry : faces) {
            if (query_embedding.size() != entry.embedding.size()) {
                continue;
            }

            // Cosine similarity
            float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
            for (size_t i = 0; i < query_embedding.size(); i++) {
                dot += query_embedding[i] * entry.embedding[i];
                norm_a += query_embedding[i] * query_embedding[i];
                norm_b += entry.embedding[i] * entry.embedding[i];
            }
            float sim = dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-6f);

            if (person_max_sim.find(entry.name) == person_max_sim.end() || 
                sim > person_max_sim[entry.name]) {
                person_max_sim[entry.name] = sim;
            }
        }

        if (person_max_sim.empty()) {
            return {"Unknown", 0.0f, false};
        }

        // Sort by similarity
        std::vector<std::pair<std::string, float>> similarities;
        for (const auto& [name, sim] : person_max_sim) {
            similarities.push_back({name, sim});
        }
        std::sort(similarities.begin(), similarities.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        std::string best_match = similarities[0].first;
        float best_sim = similarities[0].second;

        if (best_sim < threshold_) {
            return {"Unknown", best_sim, false};
        }

        if (similarities.size() >= 2) {
            float second_sim = similarities[1].second;
            float gap = best_sim - second_sim;
            if (gap < min_gap_ && second_sim > threshold_) {
                return {"Unknown", best_sim, false};
            }
        }

        return {best_match, best_sim, true};
    }

    /**
     * @brief Get all registered persons
     */
    std::vector<PersonInfo> get_all_persons() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<PersonInfo> result;
        
        if (!conn_) return result;

        std::string sql = "SELECT subject, COUNT(*) as cnt FROM " + table_name_ + " GROUP BY subject";
        
        if (mysql_query(conn_, sql.c_str()) != 0) {
            return result;
        }

        MYSQL_RES* res = mysql_store_result(conn_);
        if (!res) return result;

        MYSQL_ROW row;
        while ((row = mysql_fetch_row(res))) {
            PersonInfo info;
            info.name = row[0] ? row[0] : "";
            info.embedding_count = row[1] ? std::stoi(row[1]) : 0;
            result.push_back(info);
        }

        mysql_free_result(res);
        return result;
    }

    size_t person_count() {
        auto persons = get_all_persons();
        return persons.size();
    }

    size_t embedding_count() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!conn_) return 0;

        std::string sql = "SELECT COUNT(*) FROM " + table_name_;
        if (mysql_query(conn_, sql.c_str()) != 0) return 0;

        MYSQL_RES* res = mysql_store_result(conn_);
        if (!res) return 0;

        MYSQL_ROW row = mysql_fetch_row(res);
        size_t count = row && row[0] ? std::stoul(row[0]) : 0;
        mysql_free_result(res);
        return count;
    }

    // No-op for MySQL (always persisted)
    bool save() { return true; }
    bool load(const std::string&) { return is_connected(); }

    void set_threshold(float threshold) { threshold_ = threshold; }
    void set_min_gap(float min_gap) { min_gap_ = min_gap; }
};
