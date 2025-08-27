#include "../../include/core/IAnalysisModule.h"
#include "../../include/core/AudioBuffer.h"
#include <random>
#include <cmath>

namespace ave::modules {

/**
 * FAKE Onset Detection Module
 */
class FakeOnsetModule : public core::IAnalysisModule {
private:
    float m_sensitivity = 0.5f;
    
public:
    std::string getName() const override { return "Onset"; }
    std::string getVersion() const override { return "1.0.0-fake"; }
    
    bool initialize(const nlohmann::json& config) override {
        if (config.contains("sensitivity")) {
            m_sensitivity = config["sensitivity"];
        }
        return true;
    }
    
    void reset() override {
        m_sensitivity = 0.5f;
    }
    
    nlohmann::json process(const core::AudioBuffer& audio, 
                          const core::AnalysisContext& context) override {
        // Generate fake onsets based on BPM if available
        nlohmann::json onsets = nlohmann::json::array();
        
        auto bpmResult = context.getModuleResult("BPM");
        if (bpmResult && bpmResult->contains("beatGrid")) {
            // Add onsets at beat positions with some variation
            auto& beatGrid = (*bpmResult)["beatGrid"];
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> offsetDist(-0.02, 0.02);
            
            for (auto& beat : beatGrid) {
                float beatTime = beat["t"];
                onsets.push_back({
                    {"t", beatTime + offsetDist(gen)},
                    {"strength", 0.7f + offsetDist(gen) * 5},
                    {"type", "percussive"}
                });
            }
        }
        
        return {
            {"onsets", onsets},
            {"count", onsets.size()},
            {"sensitivity", m_sensitivity}
        };
    }
    
    bool validateOutput(const nlohmann::json& output) const override {
        return output.contains("onsets") && output.contains("count");
    }
    
    std::vector<std::string> getDependencies() const override {
        return {"BPM"}; // Depends on BPM for better fake data
    }
};

/**
 * FAKE Structure/Segmentation Module  
 */
class FakeStructureModule : public core::IAnalysisModule {
private:
    float m_segmentMinLength = 8.0f;
    
public:
    std::string getName() const override { return "Structure"; }
    std::string getVersion() const override { return "1.0.0-fake"; }
    
    bool initialize(const nlohmann::json& config) override {
        if (config.contains("segmentMinLength")) {
            m_segmentMinLength = config["segmentMinLength"];
        }
        return true;
    }
    
    void reset() override {
        m_segmentMinLength = 8.0f;
    }
    
    nlohmann::json process(const core::AudioBuffer& audio, 
                          const core::AnalysisContext& context) override {
        // Generate fake song structure
        nlohmann::json segments = nlohmann::json::array();
        float duration = audio.getDuration();
        
        // Typical EDM structure (fake)
        std::vector<std::pair<std::string, float>> structure = {
            {"intro", 0.1f},
            {"buildup", 0.15f},
            {"drop", 0.25f},
            {"breakdown", 0.15f},
            {"buildup", 0.1f},
            {"drop", 0.2f},
            {"outro", 0.05f}
        };
        
        float currentTime = 0.0f;
        for (auto& [label, ratio] : structure) {
            float segmentDuration = duration * ratio;
            if (segmentDuration < m_segmentMinLength && label != "outro") {
                segmentDuration = m_segmentMinLength;
            }
            
            segments.push_back({
                {"start", currentTime},
                {"end", std::min(currentTime + segmentDuration, duration)},
                {"label", label},
                {"confidence", 0.75f}
            });
            
            currentTime += segmentDuration;
            if (currentTime >= duration) break;
        }
        
        return {
            {"segments", segments},
            {"count", segments.size()}
        };
    }
    
    bool validateOutput(const nlohmann::json& output) const override {
        return output.contains("segments");
    }
};

/**
 * FAKE Tonality/Key Detection Module
 */
class FakeTonalityModule : public core::IAnalysisModule {
public:
    std::string getName() const override { return "Tonality"; }
    std::string getVersion() const override { return "1.0.0-fake"; }
    
    bool initialize(const nlohmann::json& config) override {
        return true;
    }
    
    void reset() override {}
    
    nlohmann::json process(const core::AudioBuffer& audio, 
                          const core::AnalysisContext& context) override {
        // Generate fake key detection
        std::vector<std::string> keys = {
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
        };
        std::vector<std::string> modes = {"major", "minor"};
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> keyDist(0, keys.size() - 1);
        std::uniform_int_distribution<> modeDist(0, 1);
        std::uniform_real_distribution<> confDist(0.6, 0.95);
        
        std::string key = keys[keyDist(gen)];
        std::string mode = modes[modeDist(gen)];
        
        return {
            {"key", key},
            {"mode", mode},
            {"keyString", key + (mode == "minor" ? "m" : "")},
            {"confidence", confDist(gen)},
            {"chromaVector", std::vector<float>(12, 0.1f)} // Fake chroma
        };
    }
    
    bool validateOutput(const nlohmann::json& output) const override {
        return output.contains("key") && output.contains("mode");
    }
};

/**
 * FAKE Spectral Analysis Module
 */
class FakeSpectralModule : public core::IAnalysisModule {
private:
    size_t m_fftSize = 2048;
    size_t m_hopSize = 512;
    
public:
    std::string getName() const override { return "Spectral"; }
    std::string getVersion() const override { return "1.0.0-fake"; }
    
    bool initialize(const nlohmann::json& config) override {
        if (config.contains("fftSize")) {
            m_fftSize = config["fftSize"];
        }
        if (config.contains("hopSize")) {
            m_hopSize = config["hopSize"];
        }
        return true;
    }
    
    void reset() override {
        m_fftSize = 2048;
        m_hopSize = 512;
    }
    
    nlohmann::json process(const core::AudioBuffer& audio, 
                          const core::AnalysisContext& context) override {
        // Generate fake spectral bands over time
        float duration = audio.getDuration();
        float frameRate = audio.getSampleRate() / m_hopSize;
        size_t numFrames = static_cast<size_t>(duration * frameRate);
        
        // Generate fake energy curves for frequency bands
        auto generateBand = [numFrames](float baseLevel, float variation) {
            nlohmann::json band = nlohmann::json::array();
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dist(-variation, variation);
            
            for (size_t i = 0; i < numFrames; i += 10) { // Subsample
                float t = i / static_cast<float>(numFrames);
                float envelope = std::sin(t * M_PI) * 0.3f + 0.7f; // Fake envelope
                band.push_back({
                    {"t", t},
                    {"v", std::max<float>(0.0f, baseLevel * envelope + dist(gen))}
                });
            }
            return band;
        };
        
        return {
            {"bands", {
                {"low", generateBand(0.7f, 0.1f)},     // 0-250 Hz
                {"lowMid", generateBand(0.5f, 0.1f)},  // 250-500 Hz
                {"mid", generateBand(0.6f, 0.15f)},    // 500-2000 Hz
                {"highMid", generateBand(0.4f, 0.1f)}, // 2000-4000 Hz
                {"high", generateBand(0.3f, 0.1f)}     // 4000+ Hz
            }},
            {"fftSize", m_fftSize},
            {"hopSize", m_hopSize},
            {"frameRate", frameRate}
        };
    }
    
    bool validateOutput(const nlohmann::json& output) const override {
        return output.contains("bands") && output.contains("frameRate");
    }
};

/**
 * FAKE Cue Detection Module  
 * Depends on multiple other modules
 */
class FakeCueModule : public core::IAnalysisModule {
private:
    float m_anticipationTime = 1.0f;
    
public:
    std::string getName() const override { return "Cue"; }
    std::string getVersion() const override { return "1.0.0-fake"; }
    
    bool initialize(const nlohmann::json& config) override {
        if (config.contains("anticipationTime")) {
            m_anticipationTime = config["anticipationTime"];
        }
        return true;
    }
    
    void reset() override {
        m_anticipationTime = 1.0f;
    }
    
    nlohmann::json process(const core::AudioBuffer& audio, 
                          const core::AnalysisContext& context) override {
        nlohmann::json cues = nlohmann::json::array();
        
        // Generate cues based on structure
        auto structResult = context.getModuleResult("Structure");
        if (structResult && structResult->contains("segments")) {
            for (auto& segment : (*structResult)["segments"]) {
                std::string label = segment["label"];
                float startTime = segment["start"];
                
                if (label == "drop") {
                    // Add anticipation cue before drop
                    if (startTime > m_anticipationTime) {
                        cues.push_back({
                            {"t", startTime - m_anticipationTime},
                            {"type", "pre-drop"},
                            {"strength", 0.8f},
                            {"duration", m_anticipationTime}
                        });
                    }
                    // Add drop cue
                    cues.push_back({
                        {"t", startTime},
                        {"type", "drop"},
                        {"strength", 1.0f},
                        {"duration", 0.0f}
                    });
                } else if (label == "buildup") {
                    cues.push_back({
                        {"t", startTime},
                        {"type", "buildup"},
                        {"strength", 0.6f},
                        {"duration", segment["end"].get<float>() - startTime}
                    });
                }
            }
        }
        
        return {
            {"cues", cues},
            {"count", cues.size()},
            {"anticipationTime", m_anticipationTime}
        };
    }
    
    bool validateOutput(const nlohmann::json& output) const override {
        return output.contains("cues");
    }
    
    std::vector<std::string> getDependencies() const override {
        return {"Structure", "BPM"};
    }
};

// Factory functions
std::unique_ptr<core::IAnalysisModule> createFakeOnsetModule() {
    return std::make_unique<FakeOnsetModule>();
}

std::unique_ptr<core::IAnalysisModule> createFakeStructureModule() {
    return std::make_unique<FakeStructureModule>();
}

std::unique_ptr<core::IAnalysisModule> createFakeTonalityModule() {
    return std::make_unique<FakeTonalityModule>();
}

std::unique_ptr<core::IAnalysisModule> createFakeSpectralModule() {
    return std::make_unique<FakeSpectralModule>();
}

std::unique_ptr<core::IAnalysisModule> createFakeCueModule() {
    return std::make_unique<FakeCueModule>();
}

} // namespace ave::modules

