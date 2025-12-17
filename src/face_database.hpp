#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <iomanip>
#include <random>
#include <chrono>
#include <iostream>

/**
 * @brief Face Database for storing and matching face embeddings
 * 
 * Stores embeddings in text file format: image_id|name|base64|embedding1,embedding2,...
 * Supports multiple embeddings per person for better recognition accuracy.
 */
class FaceDatabase {
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
    // Store all face entries
    std::vector<FaceEntry> entries_;
    std::set<std::string> unique_names_;
    std::string database_path_;
    mutable std::mutex mutex_;
    
    // Recognition parameters
    float threshold_ = 0.4f;      // Cosine similarity threshold
    float min_gap_ = 0.08f;       // Minimum gap between top-1 and top-2
    bool enable_gap_logging_ = true;
    
    struct ConfidenceGapInfo {
        std::string timestamp;
        std::string top1_name;
        float top1_similarity;
        std::string top2_name;
        float top2_similarity;
        float gap;
        float min_gap_threshold;
        float similarity_threshold;
        bool rejected;
        std::string rejection_reason;
    };
    std::vector<ConfidenceGapInfo> gap_log_;

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

public:
    FaceDatabase() = default;
    explicit FaceDatabase(const std::string& path) : database_path_(path) {
        load(path);
    }

    /**
     * @brief Load database from file
     * Format: image_id|name|base64|embedding1,embedding2,...
     */
    bool load(const std::string& path) {
        std::lock_guard<std::mutex> lock(mutex_);
        database_path_ = path;
        entries_.clear();
        unique_names_.clear();

        std::ifstream file(path);
        if (!file.is_open()) {
            // File doesn't exist yet, that's OK
            return true;
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;

            // Parse format: image_id|name|base64|embeddings
            std::vector<std::string> parts;
            std::stringstream ss(line);
            std::string part;
            while (std::getline(ss, part, '|')) {
                parts.push_back(part);
            }

            // Support old format (name|embeddings) and new format (image_id|name|base64|embeddings)
            FaceEntry entry;
            if (parts.size() == 2) {
                // Old format: name|embeddings
                entry.image_id = generate_uuid();
                entry.name = parts[0];
                entry.base64_image = "";
                
                std::stringstream emb_ss(parts[1]);
                std::string value;
                while (std::getline(emb_ss, value, ',')) {
                    try {
                        entry.embedding.push_back(std::stof(value));
                    } catch (...) {
                        continue;
                    }
                }
            } else if (parts.size() >= 4) {
                // New format: image_id|name|base64|embeddings
                entry.image_id = parts[0];
                entry.name = parts[1];
                entry.base64_image = parts[2];
                
                std::stringstream emb_ss(parts[3]);
                std::string value;
                while (std::getline(emb_ss, value, ',')) {
                    try {
                        entry.embedding.push_back(std::stof(value));
                    } catch (...) {
                        continue;
                    }
                }
            } else {
                continue;
            }

            if (!entry.embedding.empty()) {
                entries_.push_back(entry);
                unique_names_.insert(entry.name);
            }
        }

        return true;
    }

    /**
     * @brief Save database to file
     * Format: image_id|name|base64|embedding1,embedding2,...
     */
    bool save() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::ofstream file(database_path_);
        if (!file.is_open()) {
            return false;
        }

        for (const auto& entry : entries_) {
            file << entry.image_id << "|" << entry.name << "|" << entry.base64_image << "|";
            for (size_t i = 0; i < entry.embedding.size(); i++) {
                file << std::fixed << std::setprecision(6) << entry.embedding[i];
                if (i < entry.embedding.size() - 1) file << ",";
            }
            file << "\n";
        }

        return true;
    }

    /**
     * @brief Add a new face entry with base64 image
     */
    std::string add_face(const std::string& name, const std::vector<float>& embedding, 
                         const std::string& base64_image = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        
        FaceEntry entry;
        entry.image_id = generate_uuid();
        entry.name = name;
        entry.base64_image = base64_image;
        entry.embedding = embedding;
        
        entries_.push_back(entry);
        unique_names_.insert(name);
        
        return entry.image_id;
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
        
        auto it = unique_names_.find(name);
        if (it == unique_names_.end()) {
            return false;
        }

        // Remove all entries for this person
        entries_.erase(
            std::remove_if(entries_.begin(), entries_.end(),
                [&name](const auto& entry) { return entry.name == name; }),
            entries_.end()
        );
        
        unique_names_.erase(it);
        return true;
    }

    /**
     * @brief Rename a person (all their entries)
     */
    bool rename_person(const std::string& old_name, const std::string& new_name) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Check old name exists
        auto it = unique_names_.find(old_name);
        if (it == unique_names_.end()) {
            return false;
        }

        // Update all entries with old name
        for (auto& entry : entries_) {
            if (entry.name == old_name) {
                entry.name = new_name;
            }
        }
        
        // Update unique names set
        unique_names_.erase(it);
        unique_names_.insert(new_name);
        return true;
    }

    /**
     * @brief Get all face entries
     */
    std::vector<FaceEntry> get_all_faces() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_;
    }

    /**
     * @brief Identify a face by its embedding
     */
    IdentifyResult identify(const std::vector<float>& query_embedding) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (query_embedding.empty() || entries_.empty()) {
            return {"Unknown", 0.0f, false};
        }

        // Calculate max similarity for each person
        std::map<std::string, float> person_max_sim;

        for (const auto& entry : entries_) {
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

            // Keep max similarity per person
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

        // Check threshold
        if (best_sim < threshold_) {
            log_confidence_gap(best_match, best_sim,
                similarities.size() >= 2 ? similarities[1].first : "",
                similarities.size() >= 2 ? similarities[1].second : 0.0f,
                true, "below_threshold");
            return {"Unknown", best_sim, false};
        }

        // Confidence gap check
        if (similarities.size() >= 2) {
            float second_sim = similarities[1].second;
            float gap = best_sim - second_sim;
            
            // Log confidence gap info - reject if gap is below min_gap
            bool rejected = (gap < min_gap_);
            log_confidence_gap(best_match, best_sim,
                similarities[1].first, second_sim,
                rejected, rejected ? "insufficient_gap" : "passed");
            
            if (rejected) {
                // Ambiguous match - gap too small
                return {"Unknown", best_sim, false};
            }
        } else {
            // Only one person in database
            log_confidence_gap(best_match, best_sim, "", 0.0f, false, "single_person");
        }

        return {best_match, best_sim, true};
    }

    /**
     * @brief Get all registered persons
     */
    std::vector<PersonInfo> get_all_persons() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::map<std::string, int> counts;
        for (const auto& entry : entries_) {
            counts[entry.name]++;
        }

        std::vector<PersonInfo> result;
        for (const auto& [name, count] : counts) {
            result.push_back({name, count});
        }
        return result;
    }

    /**
     * @brief Get list of unique names
     */
    std::vector<std::string> get_names() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return std::vector<std::string>(unique_names_.begin(), unique_names_.end());
    }

    size_t person_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return unique_names_.size();
    }

    size_t embedding_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.size();
    }

    int embedding_dim() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (entries_.empty()) return 0;
        return static_cast<int>(entries_[0].embedding.size());
    }

    void set_threshold(float threshold) { threshold_ = threshold; }
    void set_min_gap(float min_gap) { min_gap_ = min_gap; }
    void set_gap_logging(bool enable) { enable_gap_logging_ = enable; }

    /**
     * @brief Get current timestamp as string
     */
    static std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
        ss << "." << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }

    /**
     * @brief Log confidence gap information
     */
    void log_confidence_gap(const std::string& top1_name, float top1_sim,
                           const std::string& top2_name, float top2_sim,
                           bool rejected, const std::string& reason) const {
        if (!enable_gap_logging_) return;
        
        float gap = top1_sim - top2_sim;
        
        ConfidenceGapInfo info;
        info.timestamp = get_timestamp();
        info.top1_name = top1_name;
        info.top1_similarity = top1_sim;
        info.top2_name = top2_name;
        info.top2_similarity = top2_sim;
        info.gap = gap;
        info.min_gap_threshold = min_gap_;
        info.similarity_threshold = threshold_;
        info.rejected = rejected;
        info.rejection_reason = reason;
        
        const_cast<std::vector<ConfidenceGapInfo>&>(gap_log_).push_back(info);
        
        // Print detailed log
        std::cerr << "[CONFIDENCE_GAP] " << info.timestamp << " | "
                  << "Top1: " << top1_name << " (" << std::fixed << std::setprecision(4) << top1_sim << ") | "
                  << "Top2: " << top2_name << " (" << std::fixed << std::setprecision(4) << top2_sim << ") | "
                  << "Gap: " << std::fixed << std::setprecision(4) << gap << " (min: " << min_gap_ << ") | "
                  << "Result: " << (rejected ? "REJECTED" : "ACCEPTED") << " (" << reason << ")"
                  << std::endl;
    }

    /**
     * @brief Get all confidence gap logs
     */
    std::vector<ConfidenceGapInfo> get_gap_logs() const {
        return gap_log_;
    }

    /**
     * @brief Get only rejected confidence gap logs
     */
    std::vector<ConfidenceGapInfo> get_rejected_gap_logs() const {
        std::vector<ConfidenceGapInfo> result;
        for (const auto& info : gap_log_) {
            if (info.rejected) {
                result.push_back(info);
            }
        }
        return result;
    }

    /**
     * @brief Clear confidence gap logs
     */
    void clear_gap_logs() {
        gap_log_.clear();
    }

    /**
     * @brief Print confidence gap summary to stderr
     */
    void print_gap_summary() const {
        int total = gap_log_.size();
        int rejected = 0;
        int below_threshold = 0;
        int insufficient_gap = 0;
        
        for (const auto& info : gap_log_) {
            if (info.rejected) {
                rejected++;
                if (info.rejection_reason == "below_threshold") below_threshold++;
                if (info.rejection_reason == "insufficient_gap") insufficient_gap++;
            }
        }
        
        std::cerr << "\n====== CONFIDENCE GAP SUMMARY ======" << std::endl;
        std::cerr << "Total recognitions:      " << total << std::endl;
        std::cerr << "Accepted:                " << (total - rejected) << std::endl;
        std::cerr << "Rejected:                " << rejected << std::endl;
        std::cerr << "  - Below threshold:     " << below_threshold << std::endl;
        std::cerr << "  - Insufficient gap:    " << insufficient_gap << std::endl;
        std::cerr << "Current threshold:       " << threshold_ << std::endl;
        std::cerr << "Current min_gap:         " << min_gap_ << std::endl;
        std::cerr << "====================================\n" << std::endl;
    }
};
