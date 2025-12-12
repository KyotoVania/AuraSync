#include "../../include/pipeline/AnalysisPipeline.h"
#include "../../include/core/JsonContract.h"
#include <iostream>
#include <queue>
#include <set>
#include <algorithm>

namespace ave::pipeline {

/**
 * @brief Constructs the AnalysisPipeline and initializes the global configuration.
 */
AnalysisPipeline::AnalysisPipeline()
    : m_globalConfig({
        {"version", 1},
        {"timestamp", ""},
        {"debug", false}
    })
{
}

/**
 * @brief Default destructor for the AnalysisPipeline.
 */
AnalysisPipeline::~AnalysisPipeline() = default;

/**
 * @brief Registers an analysis module with the pipeline.
 *
 * The module's dependencies are extracted, and it is initially enabled.
 * @param module A unique pointer to the IAnalysisModule instance.
 */
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

/**
 * @brief Registers a factory capable of creating an analysis module.
 * @param factory A unique pointer to the IModuleFactory instance.
 */
void AnalysisPipeline::registerFactory(std::unique_ptr<core::IModuleFactory> factory) {
    if (!factory) return;

    std::string name = factory->getModuleName();
    m_factories[name] = std::move(factory);
}

/**
 * @brief Enables or disables a specific analysis module by name.
 * @param name The name of the module.
 * @param enabled true to enable, false to disable.
 */
void AnalysisPipeline::enableModule(const std::string& name, bool enabled) {
    if (m_modules.count(name)) {
        m_modules[name].enabled = enabled;
    }
}

/**
 * @brief Checks if a specific module is currently enabled.
 * @param name The name of the module.
 * @return true if the module exists and is enabled, false otherwise.
 */
bool AnalysisPipeline::isModuleEnabled(const std::string& name) const {
    auto it = m_modules.find(name);
    return it != m_modules.end() && it->second.enabled;
}

/**
 * @brief Retrieves a list of all registered module names.
 * @return A vector of strings containing module names.
 */
std::vector<std::string> AnalysisPipeline::getModuleNames() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : m_modules) {
        names.push_back(name);
    }
    return names;
}

/**
 * @brief Sets or updates the global configuration using JSON merge-patch.
 * @param config The JSON object containing global configuration parameters.
 */
void AnalysisPipeline::setGlobalConfig(const nlohmann::json& config) {
    m_globalConfig.merge_patch(config);
}

/**
 * @brief Sets the specific configuration for a registered module.
 * @param moduleName The name of the module to configure.
 * @param config The JSON object containing the module's configuration.
 */
void AnalysisPipeline::setModuleConfig(const std::string& moduleName,
                                        const nlohmann::json& config) {
    if (m_modules.count(moduleName)) {
        m_modules[moduleName].config = config;
    }
}

/**
 * @brief Executes the full analysis pipeline on the provided audio buffer.
 *
 * Modules are executed in dependency order (topological sort).
 * @param audio The input AudioBuffer to analyze.
 * @param progress An optional callback function to report analysis progress.
 * @return A JSON object containing the combined analysis results from all executed modules,
 * formatted by the JsonContract.
 * @throw std::runtime_error if initialization fails or a circular dependency is detected.
 */
nlohmann::json AnalysisPipeline::analyze(const core::AudioBuffer& audio,
                                         ProgressCallback progress) {
    std::cout << "[Pipeline] Starting analysis..." << std::endl;

    // Initialize context
    core::AnalysisContext context;
    context.sampleRate = audio.getSampleRate();
    context.globalConfig = m_globalConfig;

    // Get execution order based on dependencies
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
        // Store the result for use by subsequent dependent modules
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

/**
 * @brief Resets the internal state of all registered modules.
 */
void AnalysisPipeline::reset() {
    for (auto& [name, info] : m_modules) {
        if (info.module) {
            info.module->reset();
        }
    }
}

/**
 * @brief Clears all registered modules and factories from the pipeline.
 */
void AnalysisPipeline::clear() {
    m_modules.clear();
    m_factories.clear();
}

/**
 * @brief Retrieves the topologically sorted execution order of enabled modules.
 * @return A vector of module names in the order they should be executed.
 */
std::vector<std::string> AnalysisPipeline::getExecutionOrder() const {
    return topologicalSort();
}

/**
 * @brief Validates if the current dependencies among enabled modules are valid (i.e., no cycles).
 * @return true if dependencies are valid (no cycles), false otherwise.
 */
bool AnalysisPipeline::validateDependencies() const {
    return !hasCycles();
}

/**
 * @brief Performs a topological sort on the enabled modules to determine the execution order.
 *
 * This implementation uses Kahn's algorithm (based on in-degrees and a queue).
 * @return A vector of module names in topological order, or an empty vector if a cycle is detected.
 */
std::vector<std::string> AnalysisPipeline::topologicalSort() const {
    std::vector<std::string> result;
    // Map storing the number of unmet dependencies for each module
    std::map<std::string, size_t> inDegree;
    // Adjacency list: A -> [B, C] means A must run before B and C
    std::map<std::string, std::vector<std::string>> adjacency;

    // Build adjacency list and calculate initial in-degrees
    for (const auto& [name, info] : m_modules) {
        if (!info.enabled) continue;

        inDegree[name] = info.dependencies.size();

        for (const auto& dep : info.dependencies) {
            if (m_modules.count(dep) && m_modules.at(dep).enabled) {
                // Dependency 'dep' must run before 'name'
                adjacency[dep].push_back(name);
            } else if (m_modules.count(dep)) {
                // Dependency exists but is disabled, so we count it as met
                inDegree[name]--;
            }
        }
    }

    // Queue for modules with no unmet dependencies (in-degree == 0)
    std::queue<std::string> queue;
    for (const auto& [name, degree] : inDegree) {
        if (degree == 0) {
            queue.push(name);
        }
    }

    // Process queue (Kahn's algorithm)
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

    // Check if all modules were processed (if result size != inDegree size, a cycle was detected)
    if (result.size() != inDegree.size()) {
        // Cycle detected
        return {};
    }

    return result;
}

/**
 * @brief Checks if a dependency cycle exists among the enabled modules.
 * @return true if a cycle is detected, false otherwise.
 */
bool AnalysisPipeline::hasCycles() const {
    return getExecutionOrder().empty() && !m_modules.empty();
}

/**
 * @brief Executes a single module and validates its output.
 * @param name The name of the module to execute.
 * @param audio The input AudioBuffer.
 * @param context The current analysis context, including results from previous modules.
 * @return The JSON result generated by the module.
 * @throw std::runtime_error if module output validation fails.
 */
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

/**
 * @brief Starts the builder or adds configuration for the BPM module.
 * @param minBPM The minimum expected BPM.
 * @param maxBPM The maximum expected BPM.
 * @return Reference to the builder for chaining.
 */
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

/**
 * @brief Starts the builder or adds configuration for the Onset detection module.
 * @param sensitivity The sensitivity level for onset detection.
 * @return Reference to the builder for chaining.
 */
PipelineBuilder& PipelineBuilder::withOnsets(float sensitivity) {
    if (!m_pipeline) {
        m_pipeline = std::make_unique<AnalysisPipeline>();
    }

    m_config["Onset"] = {
        {"sensitivity", sensitivity}
    };

    return *this;
}

/**
 * @brief Starts the builder or adds configuration for the Structure analysis module.
 * @param segmentMinLength The minimum length required for a structural segment.
 * @return Reference to the builder for chaining.
 */
PipelineBuilder& PipelineBuilder::withStructure(float segmentMinLength) {
    if (!m_pipeline) {
        m_pipeline = std::make_unique<AnalysisPipeline>();
    }

    m_config["Structure"] = {
        {"segmentMinLength", segmentMinLength}
    };

    return *this;
}

/**
 * @brief Starts the builder or adds configuration for the Tonality module.
 * @param includeChords Flag to indicate if chord estimation should be performed.
 * @return Reference to the builder for chaining.
 */
PipelineBuilder& PipelineBuilder::withTonality(bool includeChords) {
    if (!m_pipeline) {
        m_pipeline = std::make_unique<AnalysisPipeline>();
    }

    m_config["Tonality"] = {
        {"includeChords", includeChords}
    };

    return *this;
}

/**
 * @brief Starts the builder or adds configuration for the Spectral features module.
 * @param fftSize The size of the FFT window. Hop size is automatically set to fftSize / 4.
 * @return Reference to the builder for chaining.
 */
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

/**
 * @brief Starts the builder or adds configuration for the Cue module.
 * @param anticipationTime The time before an event to anticipate a cue.
 * @return Reference to the builder for chaining.
 */
PipelineBuilder& PipelineBuilder::withCues(float anticipationTime) {
    if (!m_pipeline) {
        m_pipeline = std::make_unique<AnalysisPipeline>();
    }

    m_config["Cue"] = {
        {"anticipationTime", anticipationTime}
    };

    return *this;
}

/**
 * @brief Manually registers a pre-instantiated module with the builder's pipeline.
 * @param module A unique pointer to the IAnalysisModule instance.
 * @return Reference to the builder for chaining.
 */
PipelineBuilder& PipelineBuilder::withModule(std::unique_ptr<core::IAnalysisModule> module) {
    if (!m_pipeline) {
        m_pipeline = std::make_unique<AnalysisPipeline>();
    }

    m_pipeline->registerModule(std::move(module));
    return *this;
}

/**
 * @brief Merges an external JSON configuration into the module configurations.
 * @param config The JSON configuration to merge.
 * @return Reference to the builder for chaining.
 */
PipelineBuilder& PipelineBuilder::withConfig(const nlohmann::json& config) {
    m_config.merge_patch(config);
    return *this;
}

/**
 * @brief Finalizes the pipeline construction, applying all configurations.
 * @return A unique pointer to the fully configured AnalysisPipeline.
 */
std::unique_ptr<AnalysisPipeline> PipelineBuilder::build() {
    if (!m_pipeline) {
        m_pipeline = std::make_unique<AnalysisPipeline>();
    }

    // Apply configurations to the registered pipeline modules
    for (const auto& [moduleName, config] : m_config.items()) {
        m_pipeline->setModuleConfig(moduleName, config);
    }

    return std::move(m_pipeline);
}

} // namespace ave::pipeline