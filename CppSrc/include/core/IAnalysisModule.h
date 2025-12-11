#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <nlohmann/json.hpp>

namespace ave::core {

// Forward declarations
class AudioBuffer;

/**
 * @brief Context shared between different analysis modules.
 *
 * This class holds global data necessary for the analysis pipeline,
 * such as sample rate, global configuration, and results from previously
 * executed modules.
 */
class AnalysisContext {
public:
    /** @brief The sample rate of the input audio data in Hz. */
    float sampleRate = 44100.0f;

    /** @brief Stores results (as JSON) from other modules for dependency resolution. */
    std::map<std::string, nlohmann::json> moduleResults;

    /** @brief Global configuration parameters for the analysis pipeline. */
    nlohmann::json globalConfig;

    /**
     * @brief Retrieves the analysis result from a specific module.
     *
     * This is typically used by a module to access results from its dependencies.
     * @param moduleName The unique name of the module whose result is sought.
     * @return An optional containing the module's result (JSON) if found, otherwise std::nullopt.
     */
    std::optional<nlohmann::json> getModuleResult(const std::string& moduleName) const {
        auto it = moduleResults.find(moduleName);
        if (it != moduleResults.end()) {
            return it->second;
        }
        return std::nullopt;
    }
};

/**
 * @brief Base interface for all audio analysis modules.
 *
 * Each concrete module implementing this interface is responsible for
 * a single type of analysis (e.g., loudness, tempo, pitch tracking).
 */
class IAnalysisModule {
public:
    /** @brief Virtual destructor for proper inheritance cleanup. */
    virtual ~IAnalysisModule() = default;

    // Module metadata
    /**
     * @brief Returns the unique name of the module (e.g., "LoudnessExtractor").
     * @return The module's name.
     */
    virtual std::string getName() const = 0;

    /**
     * @brief Returns the version string of the module.
     * @return The module's version.
     */
    virtual std::string getVersion() const = 0;

    /**
     * @brief Indicates whether the module is designed for real-time processing.
     * @return true if the module can operate in real-time, false otherwise (default).
     */
    virtual bool isRealTime() const { return false; }

    // Lifecycle
    /**
     * @brief Initializes the module with specific configuration settings.
     *
     * This method is called once before processing starts.
     * @param config The configuration specific to this module.
     * @return true on successful initialization, false otherwise.
     */
    virtual bool initialize(const nlohmann::json& config) = 0;

    /**
     * @brief Resets the module's internal state (e.g., accumulated history).
     *
     * This is useful when the module is reused for a new file.
     */
    virtual void reset() = 0;

    // Processing
    /**
     * @brief Executes the core analysis task on the provided audio buffer.
     *
     * @param audio The input AudioBuffer to be analyzed.
     * @param context The shared AnalysisContext containing global data and dependencies.
     * @return A JSON object containing the results of the analysis.
     */
    virtual nlohmann::json process(const AudioBuffer& audio,
                                   const AnalysisContext& context) = 0;

    // Validation
    /**
     * @brief Validates the structure and content of the module's output.
     *
     * @param output The JSON object produced by the process method.
     * @return true if the output is valid, false otherwise.
     */
    virtual bool validateOutput(const nlohmann::json& output) const = 0;

    // Dependencies (for pipeline ordering)
    /**
     * @brief Lists the names of other modules whose results are required before this module can run.
     * @return A vector of module names. Returns an empty vector if there are no dependencies.
     */
    virtual std::vector<std::string> getDependencies() const {
        return {};
    }
};

/**
 * @brief Base interface for creating IAnalysisModule instances.
 *
 * This factory pattern allows the core system to create modules without
 * knowing their concrete types.
 */
class IModuleFactory {
public:
    /** @brief Virtual destructor for proper inheritance cleanup. */
    virtual ~IModuleFactory() = default;

    /**
     * @brief Creates a unique instance of the concrete IAnalysisModule.
     * @return A unique pointer to the newly created module instance.
     */
    virtual std::unique_ptr<IAnalysisModule> create() const = 0;

    /**
     * @brief Returns the name of the module produced by this factory.
     * @return The unique module name.
     */
    virtual std::string getModuleName() const = 0;
};

/**
 * @brief Concrete factory implementation for a specific module type T.
 *
 * This template simplifies the creation of factory classes for any class
 * T that inherits from IAnalysisModule.
 * @tparam T The concrete class type of the analysis module.
 */
template<typename T>
class ModuleFactory : public IModuleFactory {
public:
    /**
     * @brief Creates a unique instance of the module T.
     * @return A unique pointer to the new T instance.
     */
    std::unique_ptr<IAnalysisModule> create() const override {
        return std::make_unique<T>();
    }

    /**
     * @brief Returns the name of the module T by instantiating it temporarily.
     * @return The unique module name.
     */
    std::string getModuleName() const override {
        T instance;
        return instance.getName();
    }
};

} // namespace ave::core