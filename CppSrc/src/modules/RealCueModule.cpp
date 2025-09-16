#include "../../include/core/IAnalysisModule.h"
#include "../../include/core/AudioBuffer.h"
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace ave::modules {

/**
 * Real Cue Module - Final synthesis module that combines results from all other modules
 * to generate high-level semantic cues and enhanced segment labeling.
 */
class RealCueModule : public core::IAnalysisModule {
private:
    double m_anticipationTime = 1.5; // seconds for pre-drop cues
    
public:
    std::string getName() const override { return "Cue"; }
    std::string getVersion() const override { return "1.0.0-real"; }
    
    std::vector<std::string> getDependencies() const override {
        return {"BPM", "Onset", "Spectral", "Tonality", "Structure"};
    }
    
    bool initialize(const nlohmann::json& config) override {
        if (config.contains("anticipationTime")) {
            m_anticipationTime = config["anticipationTime"];
        }
        return true;
    }
    
    void reset() override {
        m_anticipationTime = 1.5;
    }
    
    nlohmann::json process(const core::AudioBuffer& audio, 
                          const core::AnalysisContext& context) override {
        // Task 1: Beat Phasing
        auto phasedBeats = createPhasedBeats(context);
        
        // Task 2: Analyze energy and density per segment
        auto enrichedSegments = analyzeSegmentMetrics(context);
        
        // Task 3: Apply semantic labeling to segments
        auto labeledSegments = applySemanticLabeling(enrichedSegments);
        
        // Task 4: Generate anticipation cues
        auto cues = generateCues(labeledSegments);
        
        return {
            {"segments", labeledSegments},
            {"phasedBeats", phasedBeats},
            {"cues", cues}
        };
    }
    
    bool validateOutput(const nlohmann::json& output) const override {
        return output.contains("segments") && 
               output.contains("phasedBeats") && 
               output.contains("cues");
    }
    
private:
    /**
     * Task 1: Create phased beats from BPM data
     * Adds phase information (1,2,3,4) to beat grid based on downbeats
     */
    nlohmann::json createPhasedBeats(const core::AnalysisContext& context) {
        auto bpmResult = context.getModuleResult("BPM");
        if (!bpmResult || !bpmResult->contains("beatGrid") || !bpmResult->contains("downbeats")) {
            return nlohmann::json::array();
        }
        
        auto& beatGrid = (*bpmResult)["beatGrid"];
        auto& downbeats = (*bpmResult)["downbeats"];
        
        nlohmann::json phasedBeats = nlohmann::json::array();
        int beatPhase = 1;
        size_t downbeatIndex = 0;
        
        for (const auto& beat : beatGrid) {
            float beatTime = beat["t"];
            
            // Check if this beat is a downbeat (within small tolerance)
            bool isDownbeat = false;
            while (downbeatIndex < downbeats.size()) {
                float downbeatTime = downbeats[downbeatIndex];
                if (std::abs(beatTime - downbeatTime) < 0.05f) { // 50ms tolerance
                    beatPhase = 1;
                    isDownbeat = true;
                    downbeatIndex++;
                    break;
                } else if (downbeatTime > beatTime) {
                    break;
                } else {
                    downbeatIndex++;
                }
            }
            
            phasedBeats.push_back({
                {"t", beatTime},
                {"strength", beat["strength"]},
                {"phase", beatPhase}
            });
            
            // Increment phase for next beat (unless this was a downbeat that resets to 1)
            if (!isDownbeat) {
                beatPhase = (beatPhase % 4) + 1;
            } else {
                // After a downbeat (phase 1), next beat should be phase 2
                beatPhase = 2;
            }
        }
        
        return phasedBeats;
    }
    
    /**
     * Task 2: Analyze energy and density metrics for each segment
     */
    nlohmann::json analyzeSegmentMetrics(const core::AnalysisContext& context) {
        auto structureResult = context.getModuleResult("Structure");
        auto spectralResult = context.getModuleResult("Spectral");
        auto onsetResult = context.getModuleResult("Onset");
        
        if (!structureResult || !structureResult->contains("segments")) {
            return nlohmann::json::array();
        }
        
        auto segments = (*structureResult)["segments"];
        nlohmann::json enrichedSegments = nlohmann::json::array();
        
        for (const auto& segment : segments) {
            double start = segment["start"];
            double end = segment["end"];
            double duration = end - start;
            
            // Calculate onset density
            double onsetDensity = calculateOnsetDensity(onsetResult, start, end);
            
            // Calculate spectral energy
            auto energies = calculateSpectralEnergies(spectralResult, start, end);
            
            // Create enriched segment
            nlohmann::json enriched = segment;
            enriched["duration"] = duration;
            enriched["onsetDensity"] = onsetDensity;
            enriched["lowEnergy"] = energies.low;
            enriched["midEnergy"] = energies.mid;
            enriched["highEnergy"] = energies.high;
            
            enrichedSegments.push_back(enriched);
        }
        
        return enrichedSegments;
    }
    
    /**
     * Task 3: Apply semantic labeling heuristics
     */
    nlohmann::json applySemanticLabeling(const nlohmann::json& enrichedSegments) {
        nlohmann::json labeledSegments = nlohmann::json::array();
        
        for (size_t i = 0; i < enrichedSegments.size(); ++i) {
            auto segment = enrichedSegments[i];
            double onsetDensity = segment["onsetDensity"];
            double lowEnergy = segment["lowEnergy"];
            double highEnergy = segment["highEnergy"];
            double duration = segment["duration"];
            
            std::string label = "unknown";
            
            // Heuristic rules for semantic labeling
            if (i == 0 && onsetDensity < 2.0 && lowEnergy < 0.3) {
                label = "intro";
            } else if (i == enrichedSegments.size() - 1 && duration < 20.0) {
                label = "outro";
            } else if (onsetDensity < 1.5 && lowEnergy < 0.2 && highEnergy < 0.2) {
                label = "breakdown";
            } else if (onsetDensity > 5.0 && lowEnergy > 0.7 && highEnergy > 0.4) {
                label = "drop";
            } else if (i > 0) {
                // Check for energy increase (buildup)
                double prevLowEnergy = enrichedSegments[i-1]["lowEnergy"];
                double prevHighEnergy = enrichedSegments[i-1]["highEnergy"];
                if (lowEnergy > prevLowEnergy * 1.2 && highEnergy > prevHighEnergy * 1.2) {
                    label = "buildup";
                }
            }
            
            // If no specific rule matched, keep original or assign based on energy
            if (label == "unknown") {
                if (lowEnergy > 0.5 && onsetDensity > 3.0) {
                    label = "chorus";
                } else if (lowEnergy < 0.3 && onsetDensity < 2.0) {
                    label = "verse";
                } else {
                    label = segment["label"]; // keep original
                }
            }
            
            segment["label"] = label;
            labeledSegments.push_back(segment);
        }
        
        return labeledSegments;
    }
    
    /**
     * Task 4: Generate anticipation cues and segment cues
     */
    nlohmann::json generateCues(const nlohmann::json& labeledSegments) {
        nlohmann::json cues = nlohmann::json::array();
        
        for (size_t i = 0; i < labeledSegments.size(); ++i) {
            const auto& segment = labeledSegments[i];
            std::string label = segment["label"];
            double start = segment["start"];
            double end = segment["end"];
            double duration = end - start;
            
            // Generate anticipation cues for drops
            if (label == "drop" && i > 0) {
                std::string prevLabel = labeledSegments[i-1]["label"];
                if (prevLabel != "drop") { // Beginning of a drop
                    double preDropTime = start - m_anticipationTime;
                    if (preDropTime >= 0.0) {
                        cues.push_back({
                            {"t", preDropTime},
                            {"type", "pre-drop"},
                            {"duration", m_anticipationTime}
                        });
                    }
                }
            }
            
            // Generate segment cues
            cues.push_back({
                {"t", start},
                {"type", label},
                {"duration", duration}
            });
        }
        
        return cues;
    }
    
    /**
     * Helper: Calculate onset density in a time range
     */
    double calculateOnsetDensity(const std::optional<nlohmann::json>& onsetResult, 
                                double start, double end) {
        if (!onsetResult || !onsetResult->contains("onsets")) {
            return 0.0;
        }
        
        auto& onsets = (*onsetResult)["onsets"];
        int count = 0;
        
        for (const auto& onset : onsets) {
            double t = onset["t"];
            if (t >= start && t <= end) {
                count++;
            }
        }
        
        double duration = end - start;
        return duration > 0.0 ? static_cast<double>(count) / duration : 0.0;
    }
    
    /**
     * Helper: Calculate average spectral energies in a time range
     */
    struct SpectralEnergies {
        double low = 0.0;
        double mid = 0.0;
        double high = 0.0;
    };
    
    SpectralEnergies calculateSpectralEnergies(const std::optional<nlohmann::json>& spectralResult,
                                              double start, double end) {
        SpectralEnergies energies;
        
        if (!spectralResult || !spectralResult->contains("bands")) {
            return energies;
        }
        
        auto& bands = (*spectralResult)["bands"];
        
        // Average energy for each band in the time range
        if (bands.contains("low")) {
            energies.low = calculateBandAverage(bands["low"], start, end);
        }
        if (bands.contains("mid")) {
            energies.mid = calculateBandAverage(bands["mid"], start, end);
        }
        if (bands.contains("high")) {
            energies.high = calculateBandAverage(bands["high"], start, end);
        }
        
        return energies;
    }
    
    /**
     * Helper: Calculate average value for a band in time range
     */
    double calculateBandAverage(const nlohmann::json& band, double start, double end) {
        if (!band.is_array()) {
            return 0.0;
        }
        
        double sum = 0.0;
        int count = 0;
        
        for (const auto& frame : band) {
            if (frame.contains("t") && frame.contains("v")) {
                double t = frame["t"];
                if (t >= start && t <= end) {
                    sum += static_cast<double>(frame["v"]);
                    count++;
                }
            }
        }
        
        return count > 0 ? sum / count : 0.0;
    }
};

std::unique_ptr<core::IAnalysisModule> createRealCueModule() {
    return std::make_unique<RealCueModule>();
}

} // namespace ave::modules