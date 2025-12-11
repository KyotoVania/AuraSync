#pragma once

#include "../core/IAnalysisModule.h"
#include "../core/AudioBuffer.h"
#include <memory>
#include <vector>
#include <map>
#include <functional>
#include <nlohmann/json.hpp>

namespace ave::pipeline {

/**
 * @brief Type definition for a progress reporting callback function.
 *
 * This callback is used to report the progress of long-running analysis modules.
 * @param module The name of the module currently running.
 * @param progress The current progress, typically a float between 0.0 and 1.0.
 */
using ProgressCallback = std::function<void(const std::string& module, float progress)>;

/**
 * @brief Main class managing the audio analysis pipeline.
 *
 * It is responsible for module registration, configuration, dependency resolution,
 * and sequencing the execution of all enabled analysis modules.
 */
class AnalysisPipeline {
public:
    /**
     * @brief Constructs an AnalysisPipeline instance.
     */
    AnalysisPipeline();

    /**
     * @brief Destroys the AnalysisPipeline instance, cleaning up modules and resources.
     */
    ~AnalysisPipeline();

    // Module registration
    /**
     * @brief Registers a pre-instantiated analysis module with the pipeline.
     *
     * @param module A unique pointer to the IAnalysisModule instance.
     */
    void registerModule(std::unique_ptr<core::IAnalysisModule> module);

    /**
     * @brief Registers a module factory, allowing modules to be created lazily.
     *
     * @param factory A unique pointer to the IModuleFactory instance.
     */
    void registerFactory(std::unique_ptr<core::IModuleFactory> factory);

    // Module management
    /**
     * @brief Enables or disables a registered module.
     *
     * Only enabled modules will be executed during the analysis phase.
     * @param name The unique name of the module.
     * @param enabled If true, the module is enabled; otherwise, it is disabled (default is true).
     */
    void enableModule(const std::string& name, bool enabled = true);

    /**
     * @brief Checks if a specific module is currently enabled.
     *
     * @param name The unique name of the module.
     * @return true if the module is enabled, false otherwise.
     */
    bool isModuleEnabled(const std::string& name) const;

    /**
     * @brief Retrieves a list of all registered module names.
     * @return A vector of strings containing all module names.
     */
    std::vector<std::string> getModuleNames() const;

    // Configuration
    /**
     * @brief Sets the global configuration accessible by all modules via the AnalysisContext.
     *
     * @param config The global JSON configuration object.
     */
    void setGlobalConfig(const nlohmann::json& config);

    /**
     * @brief Sets the specific configuration for a single module.
     *
     * This configuration is passed to the module's initialize method.
     * @param moduleName The name of the module to configure.
     * @param config The module's specific JSON configuration object.
     */
    void setModuleConfig(const std::string& moduleName, const nlohmann::json& config);

    // Processing
    /**
     * @brief Executes the full analysis pipeline on the provided audio buffer.
     *
     * The modules are executed in an order determined by their dependencies.
     * @param audio The input AudioBuffer containing the sound data.
     * @param progress An optional callback function to report analysis progress.
     * @return The final, structured JSON output containing all module results.
     */
    nlohmann::json analyze(const core::AudioBuffer& audio,
                           ProgressCallback progress = nullptr);

    // Pipeline control
    /**
     * @brief Resets the internal state of all registered modules.
     *
     * This is typically done before analyzing a new audio file.
     */
    void reset();

    /**
     * @brief Clears all registered modules and factories from the pipeline.
     */
    void clear();

    // Dependency resolution
    /**
     * @brief Determines the sequential order in which modules must be executed to satisfy all dependencies.
     *
     * Uses topological sorting to find the valid execution sequence.
     * @return A vector of module names in the calculated execution order.
     */
    std::vector<std::string> getExecutionOrder() const;

    /**
     * @brief Validates if all enabled modules have their dependencies registered in the pipeline.
     *
     * @return true if all dependencies are present, false otherwise.
     */
    bool validateDependencies() const;

private:
    /**
     * @brief Internal structure to hold metadata and configuration for a registered module.
     */
    struct ModuleInfo {
        std::unique_ptr<core::IAnalysisModule> module;
        nlohmann::json config;
        bool enabled = true;
        std::vector<std::string> dependencies;
    };

    std::map<std::string, ModuleInfo> m_modules;
    std::map<std::string, std::unique_ptr<core::IModuleFactory>> m_factories;
    nlohmann::json m_globalConfig;

    // Dependency resolution
    /**
     * @brief Performs a topological sort of the enabled modules based on their dependencies.
     *
     * @return A vector of module names in topological order.
     */
    std::vector<std::string> topologicalSort() const;

    /**
     * @brief Checks if the dependency graph between enabled modules contains any circular dependencies.
     *
     * @return true if a cycle is detected, false otherwise.
     */
    bool hasCycles() const;

    // Execute single module
    /**
     * @brief Initializes and executes a specific module, storing its result in the context.
     *
     * @param name The name of the module to execute.
     * @param audio The input audio buffer.
     * @param context The analysis context to store results and access dependencies.
     * @return The JSON result produced by the module.
     */
    nlohmann::json executeModule(const std::string& name,
                                 const core::AudioBuffer& audio,
                                 core::AnalysisContext& context);
};

/**
 * @brief Implements the Builder pattern for constructing and configuring an AnalysisPipeline easily.
 *
 * This simplifies the process of enabling and configuring common analysis modules.
 */
class PipelineBuilder {
public:
    /**
     * @brief Configures the BPM module with specific min/max BPM values.
     * @param minBPM The minimum BPM to consider.
     * @param maxBPM The maximum BPM to consider.
     * @return Reference to the builder for method chaining.
     */
    PipelineBuilder& withBPM(float minBPM = 60, float maxBPM = 200);

    /**
     * @brief Configures the Onset module with a specific sensitivity.
     * @param sensitivity The sensitivity threshold for onset detection.
     * @return Reference to the builder for method chaining.
     */
    PipelineBuilder& withOnsets(float sensitivity = 0.5f);

    /**
     * @brief Configures the Structure module with a minimum segment length.
     * @param segmentMinLength The minimum duration (in seconds) of a musical segment.
     * @return Reference to the builder for method chaining.
     */
    PipelineBuilder& withStructure(float segmentMinLength = 8.0f);

    /**
     * @brief Configures the Tonality module, optionally including chord detection.
     * @param includeChords If true, the tonality module should attempt chord extraction.
     * @return Reference to the builder for method chaining.
     */
    PipelineBuilder& withTonality(bool includeChords = true);

    /**
     * @brief Configures the Spectral module with the desired FFT size.
     * @param fftSize The size of the FFT window (typically a power of 2).
     * @return Reference to the builder for method chaining.
     */
    PipelineBuilder& withSpectral(size_t fftSize = 2048);

    /**
     * @brief Configures the Cue module with an anticipation time.
     * @param anticipationTime The time (in seconds) before a beat/segment boundary to place a cue.
     * @return Reference to the builder for method chaining.
     */
    PipelineBuilder& withCues(float anticipationTime = 1.0f);

    /**
     * @brief Registers a custom, pre-instantiated module.
     * @param module A unique pointer to the IAnalysisModule instance.
     * @return Reference to the builder for method chaining.
     */
    PipelineBuilder& withModule(std::unique_ptr<core::IAnalysisModule> module);

    /**
     * @brief Sets the global configuration for the pipeline.
     * @param config The global JSON configuration object.
     * @return Reference to the builder for method chaining.
     */
    PipelineBuilder& withConfig(const nlohmann::json& config);

    /**
     * @brief Finalizes the construction and returns the fully configured AnalysisPipeline.
     * @return A unique pointer to the constructed AnalysisPipeline.
     */
    std::unique_ptr<AnalysisPipeline> build();
    
private:
    std::unique_ptr<AnalysisPipeline> m_pipeline;
    nlohmann::json m_config;
};

} // namespace ave::pipeline