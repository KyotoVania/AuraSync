#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <map>
#include <cmath> // For std::abs

namespace ave::core {

/**
 * @brief Manages the JSON contract for analysis output.
 *
 * This class handles versioning, structure creation, validation, and
 * data compression for the final output JSON file.
 */
class JsonContract {
public:
    /** @brief The current version number of the JSON contract schema. */
    static constexpr int CURRENT_VERSION = 1;

    /**
     * @brief Creates the final structured JSON output from collected analysis results.
     *
     * This method aggregates results from different modules, applies compression,
     * and adds necessary metadata.
     *
     * @param audioMetadata JSON object containing basic information about the audio file (e.g., sample rate, duration).
     * @param moduleResults A map where keys are module names and values are their corresponding JSON outputs.
     * @param analysisId An optional, unique ID for the analysis. If empty, one will be generated.
     * @return The final, structured JSON analysis report.
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
                // Compress beat grid for smaller size
                {"beatGrid", compressBeatGrid(bpm["beatGrid"])},
                {"downbeats", bpm["downbeats"]}
            };
        }

        // Structure segments - prefer enhanced segments from Cue module ONLY if non-empty; otherwise fall back to Structure
        if (moduleResults.count("Cue") && moduleResults.at("Cue").contains("segments")) {
            const auto& cueSegs = moduleResults.at("Cue")["segments"];
            if ((cueSegs.is_array() && !cueSegs.empty()) || (!cueSegs.is_array() && !cueSegs.is_null())) {
                output["structure"] = cueSegs;
            } else if (moduleResults.count("Structure") && moduleResults.at("Structure").contains("segments")) {
                output["structure"] = moduleResults.at("Structure")["segments"];
            } else {
                output["structure"] = nlohmann::json::array();
            }
        } else if (moduleResults.count("Structure") && moduleResults.at("Structure").contains("segments")) {
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
            // Compress spectral bands (subsample if needed)
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
            // Placeholder: versions should be retrieved from actual modules during pipeline execution
            {"modules", getModuleVersions(moduleResults)},
            {"processingTime", 0.0} // This value will be filled by the pipeline executive
        };

        return output;
    }

    /**
     * @brief Validates the given JSON against a specified contract version.
     *
     * Currently only checks for required top-level fields for version 1.
     * @param json The JSON object to validate.
     * @param version The schema version to validate against (defaults to CURRENT_VERSION).
     * @return true if the JSON contains the required fields for the specified version, false otherwise.
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
     * @brief Compresses the beat grid data for a smaller file size.
     *
     * If the beats are regularly spaced, it stores only the start time, interval, and count.
     * Otherwise, it stores a compressed list of time and strength pairs.
     *
     * @param beatGrid The original beat grid (array of objects with 't' and 'strength').
     * @return A JSON object or array containing the compressed beat representation.
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
                // Check if the beat time is within a small tolerance (0.01s) of the expected time
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

        // Otherwise store full grid but compressed (only time and strength)
        nlohmann::json compressed = nlohmann::json::array();
        for (auto& beat : beatGrid) {
            compressed.push_back({{"t", beat["t"]}, {"s", beat.value("strength", 1.0f)}});
        }
        return compressed;
    }

    /**
     * @brief Compresses spectral bands data by subsampling if the number of points exceeds a threshold.
     *
     * This reduces the size of timeline data (e.g., energy bands over time).
     *
     * @param bands The original spectral bands (map of band name to timeline array).
     * @return A JSON object containing the compressed band timelines.
     */
    static nlohmann::json compressSpectralBands(const nlohmann::json& bands) {
        nlohmann::json compressed;

        for (auto& [name, band] : bands.items()) {
            nlohmann::json compressedBand = nlohmann::json::array();

            // Subsample if too many points (e.g., limit to 100 points)
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
     * @brief Compresses onset detection results.
     *
     * If the number of onsets is large, only the onset times ('t') are stored,
     * discarding other metadata like strength or type for reduction.
     *
     * @param onsets The original onsets array.
     * @return A JSON array containing the compressed onsets (either full objects or just times).
     */
    static nlohmann::json compressOnsets(const nlohmann::json& onsets) {
        // If few onsets, return full data
        if (onsets.size() < 50) {
            return onsets;
        }

        // Otherwise just store times
        nlohmann::json compressed = nlohmann::json::array();
        for (auto& onset : onsets) {
            compressed.push_back(onset["t"]);
        }
        return compressed;
    }

private:
    /**
     * @brief Generates the current time as an ISO 8601 formatted string (UTC).
     * @return A timestamp string in "YYYY-MM-DDTHH:MM:SSZ" format.
     */
    static std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        // Use std::gmtime instead of std::localtime for UTC time (conventionally indicated by 'Z')
        ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
        return ss.str();
    }

    /**
     * @brief Generates a simple unique analysis ID based on the current time in milliseconds.
     * @return A unique identifier string.
     */
    static std::string generateId() {
        auto now = std::chrono::steady_clock::now();
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        return "analysis_" + std::to_string(millis);
    }

    /**
     * @brief Helper to create the module version placeholder in the metadata.
     *
     * In a real pipeline, this would dynamically fetch the versions from the modules.
     * @param results The map of module results processed.
     * @return A JSON object mapping module names to a placeholder version string.
     */
    static nlohmann::json getModuleVersions(const std::map<std::string, nlohmann::json>& results) {
        nlohmann::json versions = nlohmann::json::object();
        for (auto& [name, _] : results) {
            versions[name] = "1.0.0"; // This should be filled from actual module instances during runtime
        }
        return versions;
    }
};

/**
 * @brief Constant string containing the JSON Schema for version 1 of the analysis output.
 *
 * This schema defines the expected structure and data types for the final JSON report.
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