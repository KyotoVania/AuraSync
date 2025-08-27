#include "../../include/modules/BPMModule.h"
#include "../../include/core/AudioBuffer.h"
#include "../../include/core/IAnalysisModule.h"
#include <iostream>
#include <random>
#include <algorithm> // For std::min

namespace ave::modules {

/**
 * FAKE BPM Module for testing pipeline
 * Returns simulated BPM data
 * Will be replaced by real algorithm from Rust port
 */
class FakeBPMModule : public ave::core::IAnalysisModule {
private:
    float m_minBPM = 60.0f;
    float m_maxBPM = 200.0f;
    float m_fakeBPM = 128.0f;
    float m_confidence = 0.92f;
    
public:
    std::string getName() const override {
        return "BPM";
    }
    
    std::string getVersion() const override {
        return "1.0.0-fake";
    }
    
    bool initialize(const nlohmann::json& config) override {
        std::cout << "[FakeBPM] Initializing with config: " << config.dump() << std::endl;
        
        if (config.contains("minBPM")) {
            m_minBPM = config["minBPM"];
        }
        if (config.contains("maxBPM")) {
            m_maxBPM = config["maxBPM"];
        }
        
        // Simulate BPM detection (fake)
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> bpmDist(120.0, 130.0);
        std::uniform_real_distribution<> confDist(0.85, 0.98);
        
        m_fakeBPM = bpmDist(gen);
        m_confidence = confDist(gen);
        
        return true;
    }
    
    void reset() override {
        m_fakeBPM = 128.0f;
        m_confidence = 0.92f;
    }
    
    nlohmann::json process(const ave::core::AudioBuffer& audio,
                           const ave::core::AnalysisContext& context) override {
        std::cout << "[FakeBPM] Processing " << audio.getFrameCount()
                  << " frames at " << audio.getSampleRate() << " Hz" << std::endl;
        
        // Generate fake beat grid
        nlohmann::json beatGrid = nlohmann::json::array();
        float beatInterval = 60.0f / m_fakeBPM;
        float currentTime = 0.512f; // First beat offset (fake)
        
        while (currentTime < audio.getDuration()) {
            beatGrid.push_back({
                {"t", currentTime},
                {"strength", 1.0f}
            });
            currentTime += beatInterval;
        }
        
        // Generate fake downbeats (every 4 beats)
        nlohmann::json downbeats = nlohmann::json::array();
        for (size_t i = 0; i < beatGrid.size(); i += 4) {
            if (i < beatGrid.size()) {
                downbeats.push_back(beatGrid[i]["t"]);
            }
        }
        
        // Return analysis result
        return {
            {"bpm", m_fakeBPM},
            {"confidence", m_confidence},
            {"beatInterval", beatInterval},
            {"beatGrid", beatGrid},
            {"downbeats", downbeats},
            {"method", "fake-generator"},
            {"parameters", {
                {"minBPM", m_minBPM},
                {"maxBPM", m_maxBPM}
            }}
        };
    }
    
    bool validateOutput(const nlohmann::json& output) const override {
        // Check required fields
        return output.contains("bpm") && 
               output.contains("confidence") &&
               output.contains("beatGrid") &&
               output["bpm"] >= m_minBPM &&
               output["bpm"] <= m_maxBPM;
    }
};

// Factory function
std::unique_ptr<ave::core::IAnalysisModule> createFakeBPMModule() {
    return std::make_unique<FakeBPMModule>();
}

} // namespace ave::modules