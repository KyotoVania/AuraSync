#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace ave::core {

/**
 * JSON Contract Manager
 * Handles versioning, validation, and serialization
 */
class JsonContract {
public:
    static constexpr int CURRENT_VERSION = 1;
    
    /**
     * Create the final JSON output from analysis results
     */
    static nlohmann::json createOutput(
        const nlohmann::json& audioMetadata,
        const std::map<std::string, nlohmann::json>& moduleResults,
        const std::string& analysisId = ""
    ) {
        nlohmann::json output;
        
        // Metadata
        output["version"] = CURRENT_VERSION;
        output["timestamp"] = getCurrentTimestamp();
        output["analysisId"] = analysisId.empty() ? generateId() : analysisId;
        
        // Audio metadata
        output["audio"] = audioMetadata;
        
        // Module results - organized by category
        if (moduleResults.count("BPM")) {
            auto& bpm = moduleResults.at("BPM");
            output["tempo"] = {
                {"bpm", bpm["bpm"]},
                {"confidence", bpm["confidence"]},
                {"beatGrid", compressBeatGrid(bpm["beatGrid"])},
                {"downbeats", bpm["downbeats"]}
            };
        }
        
        // Structure segments - prefer enhanced segments from Cue module
        if (moduleResults.count("Cue") && moduleResults.at("Cue").contains("segments")) {
            output["structure"] = moduleResults.at("Cue")["segments"];
        } else if (moduleResults.count("Structure")) {
            output["structure"] = moduleResults.at("Structure")["segments"];
        }
        
        if (moduleResults.count("Tonality")) {
            auto& tonality = moduleResults.at("Tonality");
            output["tonality"] = {
                {"key", tonality["keyString"]},
                {"confidence", tonality["confidence"]},
                {"chromaVector", tonality.value("chromaVector", nlohmann::json::array())}
            };
        }
        
        if (moduleResults.count("Spectral")) {
            auto& spectral = moduleResults.at("Spectral");
            output["features"]["bands"] = compressSpectralBands(spectral["bands"]);
            output["features"]["spectralInfo"] = {
                {"fftSize", spectral["fftSize"]},
                {"frameRate", spectral["frameRate"]}
            };
            if (spectral.contains("spectralTimeline")) {
                output["features"]["spectralTimeline"] = spectral["spectralTimeline"];
            }
        }
        
        if (moduleResults.count("Onset")) {
            output["features"]["onsets"] = compressOnsets(moduleResults.at("Onset")["onsets"]);
        }
        
        // Cue module outputs
        if (moduleResults.count("Cue")) {
            const auto& cueResult = moduleResults.at("Cue");
            if (cueResult.contains("cues")) {
                output["cues"] = cueResult["cues"];
            }
            if (cueResult.contains("phasedBeats")) {
                output["tempo"]["phasedBeats"] = cueResult["phasedBeats"];
            }
        }
        
        // Analysis metadata
        output["analysisMetadata"] = {
            {"modules", getModuleVersions(moduleResults)},
            {"processingTime", 0.0} // Will be filled by pipeline
        };
        
        return output;
    }
    
    /**
     * Validate JSON against schema version
     */
    static bool validate(const nlohmann::json& json, int version = CURRENT_VERSION) {
        if (!json.contains("version") || json["version"] != version) {
            return false;
        }
        
        // Check required fields for v1
        if (version == 1) {
            return json.contains("audio") &&
                   json.contains("timestamp") &&
                   json.contains("analysisId");
        }
        
        return false;
    }
    
    /**
     * Compress beat grid for smaller file size
     */
    static nlohmann::json compressBeatGrid(const nlohmann::json& beatGrid) {
        if (beatGrid.empty()) return beatGrid;
        
        // If beats are evenly spaced, just store start and interval
        std::vector<float> times;
        for (auto& beat : beatGrid) {
            times.push_back(beat["t"]);
        }
        
        if (times.size() > 2) {
            float interval = times[1] - times[0];
            bool isRegular = true;
            
            for (size_t i = 2; i < times.size(); ++i) {
                float expectedTime = times[0] + interval * i;
                if (std::abs(times[i] - expectedTime) > 0.01f) {
                    isRegular = false;
                    break;
                }
            }
            
            if (isRegular) {
                return {
                    {"type", "regular"},
                    {"start", times[0]},
                    {"interval", interval},
                    {"count", times.size()}
                };
            }
        }
        
        // Otherwise store full grid but compressed
        nlohmann::json compressed = nlohmann::json::array();
        for (auto& beat : beatGrid) {
            compressed.push_back({{"t", beat["t"]}, {"s", beat.value("strength", 1.0f)}});
        }
        return compressed;
    }
    
    /**
     * Compress spectral bands (subsample if needed)
     */
    static nlohmann::json compressSpectralBands(const nlohmann::json& bands) {
        nlohmann::json compressed;
        
        for (auto& [name, band] : bands.items()) {
            nlohmann::json compressedBand = nlohmann::json::array();
            
            // Subsample if too many points
            size_t maxPoints = 100;
            size_t bandSize = band.size();
            size_t step = (bandSize > maxPoints) ? bandSize / maxPoints : 1;
            
            for (size_t i = 0; i < bandSize; i += step) {
                compressedBand.push_back(band[i]);
            }
            
            compressed[name] = compressedBand;
        }
        
        return compressed;
    }
    
    /**
     * Compress onsets
     */
    static nlohmann::json compressOnsets(const nlohmann::json& onsets) {
        // Group onsets by type if many
        if (onsets.size() < 50) {
            return onsets;
        }
        
        // Otherwise just store times and average strength
        nlohmann::json compressed = nlohmann::json::array();
        for (auto& onset : onsets) {
            compressed.push_back(onset["t"]);
        }
        return compressed;
    }
    
private:
    static std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
        return ss.str();
    }
    
    static std::string generateId() {
        auto now = std::chrono::steady_clock::now();
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        return "analysis_" + std::to_string(millis);
    }
    
    static nlohmann::json getModuleVersions(const std::map<std::string, nlohmann::json>& results) {
        nlohmann::json versions = nlohmann::json::object();
        for (auto& [name, _] : results) {
            versions[name] = "1.0.0"; // Will be filled from actual modules
        }
        return versions;
    }
};

/**
 * JSON Schema for validation
 */
const std::string JSON_SCHEMA_V1 = R"({
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["version", "audio", "timestamp", "analysisId"],
  "properties": {
    "version": {"type": "integer", "const": 1},
    "timestamp": {"type": "string"},
    "analysisId": {"type": "string"},
    "audio": {
      "type": "object",
      "required": ["sampleRate", "duration", "channels"],
      "properties": {
        "sampleRate": {"type": "number"},
        "duration": {"type": "number"},
        "channels": {"type": "integer"}
      }
    },
    "tempo": {
      "type": "object",
      "properties": {
        "bpm": {"type": "number"},
        "confidence": {"type": "number"},
        "beatGrid": {"type": ["array", "object"]},
        "downbeats": {"type": "array"}
      }
    },
    "structure": {"type": "array"},
    "tonality": {"type": "object"},
    "features": {"type": "object"},
    "cues": {"type": "array"}
  }
})";

} // namespace ave::core