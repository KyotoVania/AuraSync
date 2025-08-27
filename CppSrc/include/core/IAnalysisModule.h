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
 * Context shared between modules
 */
class AnalysisContext {
public:
    // Sample rate from audio file
    float sampleRate = 44100.0f;
    
    // Results from other modules (for dependencies)
    std::map<std::string, nlohmann::json> moduleResults;
    
    // Global configuration
    nlohmann::json globalConfig;
    
    // Get result from another module
    std::optional<nlohmann::json> getModuleResult(const std::string& moduleName) const {
        auto it = moduleResults.find(moduleName);
        if (it != moduleResults.end()) {
            return it->second;
        }
        return std::nullopt;
    }
};

/**
 * Base interface for all analysis modules
 * Each module is responsible for one type of analysis
 */
class IAnalysisModule {
public:
    virtual ~IAnalysisModule() = default;
    
    // Module metadata
    virtual std::string getName() const = 0;
    virtual std::string getVersion() const = 0;
    virtual bool isRealTime() const { return false; }
    
    // Lifecycle
    virtual bool initialize(const nlohmann::json& config) = 0;
    virtual void reset() = 0;
    
    // Processing
    virtual nlohmann::json process(const AudioBuffer& audio, 
                                   const AnalysisContext& context) = 0;
    
    // Validation
    virtual bool validateOutput(const nlohmann::json& output) const = 0;
    
    // Dependencies (for pipeline ordering)
    virtual std::vector<std::string> getDependencies() const { 
        return {}; 
    }
};

/**
 * Factory for creating modules
 */
class IModuleFactory {
public:
    virtual ~IModuleFactory() = default;
    virtual std::unique_ptr<IAnalysisModule> create() const = 0;
    virtual std::string getModuleName() const = 0;
};

template<typename T>
class ModuleFactory : public IModuleFactory {
public:
    std::unique_ptr<IAnalysisModule> create() const override {
        return std::make_unique<T>();
    }
    
    std::string getModuleName() const override {
        T instance;
        return instance.getName();
    }
};

} // namespace ave::core