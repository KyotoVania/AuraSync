#pragma once

#include "../core/IAnalysisModule.h"
#include "../core/AudioBuffer.h"
#include <memory>
#include <vector>
#include <map>
#include <functional>

namespace ave::pipeline {

/**
 * Progress callback for long analyses
 */
using ProgressCallback = std::function<void(const std::string& module, float progress)>;

/**
 * Main analysis pipeline
 * Manages module execution order based on dependencies
 */
class AnalysisPipeline {
public:
    AnalysisPipeline();
    ~AnalysisPipeline();
    
    // Module registration
    void registerModule(std::unique_ptr<core::IAnalysisModule> module);
    void registerFactory(std::unique_ptr<core::IModuleFactory> factory);
    
    // Module management
    void enableModule(const std::string& name, bool enabled = true);
    bool isModuleEnabled(const std::string& name) const;
    std::vector<std::string> getModuleNames() const;
    
    // Configuration
    void setGlobalConfig(const nlohmann::json& config);
    void setModuleConfig(const std::string& moduleName, const nlohmann::json& config);
    
    // Processing
    nlohmann::json analyze(const core::AudioBuffer& audio, 
                           ProgressCallback progress = nullptr);
    
    // Pipeline control
    void reset();
    void clear();
    
    // Dependency resolution
    std::vector<std::string> getExecutionOrder() const;
    bool validateDependencies() const;
    
private:
    struct ModuleInfo {
        std::unique_ptr<core::IAnalysisModule> module;
        nlohmann::json config;
        bool enabled = true;
        std::vector<std::string> dependencies;
    };
    
    std::map<std::string, ModuleInfo> m_modules;
    std::map<std::string, std::unique_ptr<core::IModuleFactory>> m_factories;
    nlohmann::json m_globalConfig;
    
    // Topological sort for dependency resolution
    std::vector<std::string> topologicalSort() const;
    bool hasCycles() const;
    
    // Execute single module
    nlohmann::json executeModule(const std::string& name,
                                 const core::AudioBuffer& audio,
                                 core::AnalysisContext& context);
};

/**
 * Builder pattern for pipeline construction
 */
class PipelineBuilder {
public:
    PipelineBuilder& withBPM(float minBPM = 60, float maxBPM = 200);
    PipelineBuilder& withOnsets(float sensitivity = 0.5f);
    PipelineBuilder& withStructure(float segmentMinLength = 8.0f);
    PipelineBuilder& withTonality(bool includeChords = true);
    PipelineBuilder& withSpectral(size_t fftSize = 2048);
    PipelineBuilder& withCues(float anticipationTime = 1.0f);
    
    PipelineBuilder& withModule(std::unique_ptr<core::IAnalysisModule> module);
    PipelineBuilder& withConfig(const nlohmann::json& config);
    
    std::unique_ptr<AnalysisPipeline> build();
    
private:
    std::unique_ptr<AnalysisPipeline> m_pipeline;
    nlohmann::json m_config;
};

} // namespace ave::pipeline