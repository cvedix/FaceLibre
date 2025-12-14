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

private:
    MYSQL* conn_ = nullptr;
    std::string host_;
    int port_;
    std::string user_;
    std::string password_;
    std::string database_;
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
                      const std::string& database)
        : host_(host), port_(port), user_(user), password_(password), database_(database) {
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
     * @brief Add a new face entry
     */
    std::string add_face(const std::string& name, const std::vector<float>& embedding,
                         const std::string& base64_image = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!conn_) return "";

        std::string image_id = generate_uuid();
        std::string emb_str = serialize_embedding(embedding);
        
        std::string sql = "INSERT INTO faces (image_id, subject, base64_image, embedding) VALUES ('" +
                          escape_string(image_id) + "', '" +
                          escape_string(name) + "', '" +
                          escape_string(base64_image) + "', '" +
                          escape_string(emb_str) + "')";

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
        add_face(name, embedding, "");
    }

    /**
     * @brief Delete all entries for a person
     */
    bool delete_person(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!conn_) return false;

        std::string sql = "DELETE FROM faces WHERE subject = '" + escape_string(name) + "'";
        
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

        std::string sql = "UPDATE faces SET subject = '" + escape_string(new_name) + 
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

        std::string sql = "SELECT image_id, subject, base64_image, embedding FROM faces";
        
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

        std::string sql = "SELECT subject, COUNT(*) as cnt FROM faces GROUP BY subject";
        
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

        std::string sql = "SELECT COUNT(*) FROM faces";
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
