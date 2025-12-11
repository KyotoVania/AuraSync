#include "../../include/pipeline/AnalysisPipeline.h"
#include "../../include/core/JsonContract.h"
#include <iostream>
#include <queue>
#include <set>
#include <algorithm>

namespace ave::pipeline {

// FIX: Use member initializer list for better performance (Cppcheck recommendation)
AnalysisPipeline::AnalysisPipeline()
    : m_globalConfig({
        {"version", 1},
        {"timestamp", ""},
        {"debug", false}
    })
{
}

AnalysisPipeline::~AnalysisPipeline() = default;

void AnalysisPipeline::registerModule(std::unique_ptr<core::IAnalysisModule> module) {
    if (!module) return;
    
    std::string name = module->getName();
    ModuleInfo info;
    info.module = std::move(module);
    info.dependencies = info.module->getDependencies();
    info.enabled = true;
    info.config = nlohmann::json::object();
    
    m_modules[name] = std::move(info);
    std::cout << "[Pipeline] Registered module: " << name << std::endl;
}

void AnalysisPipeline::registerFactory(std::unique_ptr<core::IModuleFactory> factory) {
    if (!factory) return;
    
    std::string name = factory->getModuleName();
    m_factories[name] = std::move(factory);
}

void AnalysisPipeline::enableModule(const std::string& name, bool enabled) {
    if (m_modules.count(name)) {
        m_modules[name].enabled = enabled;
    }
}

bool AnalysisPipeline::isModuleEnabled(const std::string& name) const {
    auto it = m_modules.find(name);
    return it != m_modules.end() && it->second.enabled;
}

std::vector<std::string> AnalysisPipeline::getModuleNames() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : m_modules) {
        names.push_back(name);
    }
    return names;
}

void AnalysisPipeline::setGlobalConfig(const nlohmann::json& config) {
    m_globalConfig.merge_patch(config);
}

void AnalysisPipeline::setModuleConfig(const std::string& moduleName, 
                                        const nlohmann::json& config) {
    if (m_modules.count(moduleName)) {
        m_modules[moduleName].config = config;
    }
}

nlohmann::json AnalysisPipeline::analyze(const core::AudioBuffer& audio, 
                                         ProgressCallback progress) {
    std::cout << "[Pipeline] Starting analysis..." << std::endl;
    
    // Initialize context
    core::AnalysisContext context;
    context.sampleRate = audio.getSampleRate();
    context.globalConfig = m_globalConfig;
    
    // Get execution order
    auto executionOrder = getExecutionOrder();
    if (executionOrder.empty()) {
        throw std::runtime_error("No modules to execute or circular dependency detected");
    }
    
    // Initialize all modules
    for (const auto& name : executionOrder) {
        if (!isModuleEnabled(name)) continue;
        
        auto& moduleInfo = m_modules[name];
        if (!moduleInfo.module->initialize(moduleInfo.config)) {
            throw std::runtime_error("Failed to initialize module: " + name);
        }
    }
    
    // Execute modules in order
    size_t moduleIndex = 0;
    for (const auto& name : executionOrder) {
        if (!isModuleEnabled(name)) continue;
        
        if (progress) {
            float progressValue = moduleIndex / static_cast<float>(executionOrder.size());
            progress(name, progressValue);
        }
        
        std::cout << "[Pipeline] Executing: " << name << std::endl;
        
        auto result = executeModule(name, audio, context);
        context.moduleResults[name] = result;
        
        moduleIndex++;
    }
    
    if (progress) {
        progress("Complete", 1.0f);
    }
    
    // Create final output using JsonContract
    nlohmann::json audioMetadata = {
        {"sampleRate", audio.getSampleRate()},
        {"duration", audio.getDuration()},
        {"channels", audio.getChannelCount()}
    };
    
    return core::JsonContract::createOutput(audioMetadata, context.moduleResults);
}

void AnalysisPipeline::reset() {
    for (auto& [name, info] : m_modules) {
        if (info.module) {
            info.module->reset();
        }
    }
}

void AnalysisPipeline::clear() {
    m_modules.clear();
    m_factories.clear();
}

std::vector<std::string> AnalysisPipeline::getExecutionOrder() const {
    return topologicalSort();
}

bool AnalysisPipeline::validateDependencies() const {
    return !hasCycles();
}

std::vector<std::string> AnalysisPipeline::topologicalSort() const {
    std::vector<std::string> result;
    std::map<std::string, size_t> inDegree;
    std::map<std::string, std::vector<std::string>> adjacency;
    
    // Build adjacency list and calculate in-degrees
    for (const auto& [name, info] : m_modules) {
        if (!info.enabled) continue;
        
        inDegree[name] = info.dependencies.size();
        
        for (const auto& dep : info.dependencies) {
            if (m_modules.count(dep) && m_modules.at(dep).enabled) {
                adjacency[dep].push_back(name);
            } else if (m_modules.count(dep)) {
                // Dependency exists but is disabled
                inDegree[name]--;
            }
        }
    }
    
    // Queue for modules with no dependencies
    std::queue<std::string> queue;
    for (const auto& [name, degree] : inDegree) {
        if (degree == 0) {
            queue.push(name);
        }
    }
    
    // Process queue
    while (!queue.empty()) {
        std::string current = queue.front();
        queue.pop();
        result.push_back(current);
        
        // Reduce in-degree for dependent modules
        if (adjacency.count(current)) {
            for (const auto& dependent : adjacency[current]) {
                inDegree[dependent]--;
                if (inDegree[dependent] == 0) {
                    queue.push(dependent);
                }
            }
        }
    }
    
    // Check if all modules were processed (no cycles)
    if (result.size() != inDegree.size()) {
        return {}; // Cycle detected
    }
    
    return result;
}

bool AnalysisPipeline::hasCycles() const {
    return getExecutionOrder().empty() && !m_modules.empty();
}

nlohmann::json AnalysisPipeline::executeModule(const std::string& name,
                                               const core::AudioBuffer& audio,
                                               core::AnalysisContext& context) {
    auto& moduleInfo = m_modules[name];
    
    // Process
    auto result = moduleInfo.module->process(audio, context);
    
    // Validate
    if (!moduleInfo.module->validateOutput(result)) {
        throw std::runtime_error("Module output validation failed: " + name);
    }
    
    return result;
}

// ============================================================================
// PipelineBuilder Implementation
// ============================================================================

PipelineBuilder& PipelineBuilder::withBPM(float minBPM, float maxBPM) {
    if (!m_pipeline) {
        m_pipeline = std::make_unique<AnalysisPipeline>();
    }
    
    m_config["BPM"] = {
        {"minBPM", minBPM},
        {"maxBPM", maxBPM}
    };
    
    return *this;
}

PipelineBuilder& PipelineBuilder::withOnsets(float sensitivity) {
    if (!m_pipeline) {
        m_pipeline = std::make_unique<AnalysisPipeline>();
    }
    
    m_config["Onset"] = {
        {"sensitivity", sensitivity}
    };
    
    return *this;
}

PipelineBuilder& PipelineBuilder::withStructure(float segmentMinLength) {
    if (!m_pipeline) {
        m_pipeline = std::make_unique<AnalysisPipeline>();
    }
    
    m_config["Structure"] = {
        {"segmentMinLength", segmentMinLength}
    };
    
    return *this;
}

PipelineBuilder& PipelineBuilder::withTonality(bool includeChords) {
    if (!m_pipeline) {
        m_pipeline = std::make_unique<AnalysisPipeline>();
    }
    
    m_config["Tonality"] = {
        {"includeChords", includeChords}
    };
    
    return *this;
}

PipelineBuilder& PipelineBuilder::withSpectral(size_t fftSize) {
    if (!m_pipeline) {
        m_pipeline = std::make_unique<AnalysisPipeline>();
    }
    
    m_config["Spectral"] = {
        {"fftSize", fftSize},
        {"hopSize", fftSize / 4}
    };
    
    return *this;
}

PipelineBuilder& PipelineBuilder::withCues(float anticipationTime) {
    if (!m_pipeline) {
        m_pipeline = std::make_unique<AnalysisPipeline>();
    }
    
    m_config["Cue"] = {
        {"anticipationTime", anticipationTime}
    };
    
    return *this;
}

PipelineBuilder& PipelineBuilder::withModule(std::unique_ptr<core::IAnalysisModule> module) {
    if (!m_pipeline) {
        m_pipeline = std::make_unique<AnalysisPipeline>();
    }
    
    m_pipeline->registerModule(std::move(module));
    return *this;
}

PipelineBuilder& PipelineBuilder::withConfig(const nlohmann::json& config) {
    m_config.merge_patch(config);
    return *this;
}

std::unique_ptr<AnalysisPipeline> PipelineBuilder::build() {
    if (!m_pipeline) {
        m_pipeline = std::make_unique<AnalysisPipeline>();
    }
    
    // Apply configurations
    for (const auto& [moduleName, config] : m_config.items()) {
        m_pipeline->setModuleConfig(moduleName, config);
    }
    
    return std::move(m_pipeline);
}

} // namespace ave::pipeline