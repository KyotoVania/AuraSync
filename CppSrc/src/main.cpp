﻿#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>
#include <cstdlib>
#include "../include/pipeline/AnalysisPipeline.h"
#include "../include/core/AudioBuffer.h"
#include "../include/core/JsonContract.h"
#include "../include/modules/BPMModule.h"
#include "../include/modules/SpectralModule.h"
#include "../include/modules/OnsetModule.h"
#include "../include/modules/TonalityModule.h"
#include "../include/modules/StructureModule.h"
#include "../include/pipeline/AudioLoader.h"
#include <nlohmann/json.hpp>

// Forward declarations for module factories
namespace ave::modules {
    std::unique_ptr<ave::core::IAnalysisModule> createFakeOnsetModule();
    std::unique_ptr<ave::core::IAnalysisModule> createFakeStructureModule();
    std::unique_ptr<ave::core::IAnalysisModule> createFakeTonalityModule();
    std::unique_ptr<ave::core::IAnalysisModule> createFakeSpectralModule();
    std::unique_ptr<ave::core::IAnalysisModule> createRealCueModule();
}

// Real audio loader using WAV reader
ave::core::AudioBuffer loadAudioFile(const std::string& path) {
    try {
        return ave::pipeline::AudioLoader::loadWav(path);
    } catch (const std::exception& e) {
        // Configurable fallback via environment variables
        const char* envDur = std::getenv("AVE_FALLBACK_DURATION_SEC");
        const char* envCh = std::getenv("AVE_FALLBACK_CHANNELS");
        const char* envSr = std::getenv("AVE_FALLBACK_SR");
        size_t durationSec = envDur ? static_cast<size_t>(std::strtoul(envDur, nullptr, 10)) : 10;
        size_t channels = envCh ? static_cast<size_t>(std::strtoul(envCh, nullptr, 10)) : 2;
        size_t sampleRate = envSr ? static_cast<size_t>(std::strtoul(envSr, nullptr, 10)) : 44100;
        if (durationSec == 0) durationSec = 10;
        if (channels == 0) channels = 2;
        if (sampleRate == 0) sampleRate = 44100;
        std::cerr << "Failed to load WAV ('" << path << "'): " << e.what() << std::endl;
        std::cerr << "Falling back to silence buffer (configurable): duration=" << durationSec
                  << "s, channels=" << channels << ", sampleRate=" << sampleRate << std::endl;
        size_t frames = sampleRate * durationSec;
        ave::core::AudioBuffer buffer(channels, frames, static_cast<float>(sampleRate));
        return buffer;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== Audio Visual Engine - Analysis Pipeline ===" << std::endl;
    std::cout << "Version: 1.0.0-prototype" << std::endl << std::endl;

    // Parse arguments (inputPath, outputPath, configPath) - config path is required
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input.wav> <output.json> <config.json>" << std::endl;
        std::cerr << "Error: Configuration file path must be provided as the 3rd argument." << std::endl;
        return 2;
    }
    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    std::string configPath = argv[3];

    try {
        // 1. Load audio
        auto startTime = std::chrono::high_resolution_clock::now();
        ave::core::AudioBuffer audioBuffer = loadAudioFile(inputFile);

        // 2. Create and configure pipeline
        std::cout << "\nCreating analysis pipeline..." << std::endl;
        auto pipeline = std::make_unique<ave::pipeline::AnalysisPipeline>();

        // Register modules (use real BPM, others fake for now)
        std::cout << "Registering modules..." << std::endl;
        pipeline->registerModule(ave::modules::createRealBPMModule());
        pipeline->registerModule(ave::modules::createRealOnsetModule());
        pipeline->registerModule(ave::modules::createRealStructureModule());
        pipeline->registerModule(ave::modules::createRealTonalityModule());
        pipeline->registerModule(ave::modules::createRealSpectralModule());
        pipeline->registerModule(ave::modules::createRealCueModule());
        std::cout << "Modules  registered." << std::endl;
        // Load configuration file (JSON) strictly from provided path
        nlohmann::json cfg;
        {
            std::ifstream cfgIn(configPath);
            if (!cfgIn.is_open()) {
                std::cerr << "[Config] Could not open configuration file: " << configPath << std::endl;
                return 3;
            }
            try {
                cfgIn >> cfg;
                std::cout << "[Config] Loaded configuration from: " << configPath << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[Config] Failed to parse JSON ('" << configPath << "'): " << e.what() << std::endl;
                return 4;
            }
        }

        // Apply module enablement and configuration
        if (cfg.contains("modules") && cfg["modules"].is_object()) {
            for (auto& it : cfg["modules"].items()) {
                const std::string name = it.key();
                const auto& m = it.value();
                bool enabled = true;
                if (m.contains("enabled")) {
                    try { enabled = m["enabled"].get<bool>(); } catch (...) { enabled = true; }
                }
                pipeline->enableModule(name, enabled);
                std::cout << "[Config] Module '" << name << "' is " << (enabled ? "ENABLED" : "DISABLED") << std::endl;
                if (enabled && m.contains("config")) {
                    pipeline->setModuleConfig(name, m["config"]);
                    std::cout << "[Config] Applied settings to '" << name << "'" << std::endl;
                    // Extra visibility for Spectral extended mode
                    if (name == "Spectral") {
                        bool ext = false;
                        int resHz = 0;
                        try {
                            if (m["config"].contains("extendedMode")) ext = m["config"]["extendedMode"].get<bool>();
                        } catch (...) {}
                        try {
                            if (m["config"].contains("timelineResolutionHz")) resHz = m["config"]["timelineResolutionHz"].get<int>();
                        } catch (...) {}
                        std::cout << "[Config] Spectral.extendedMode=" << (ext ? "true" : "false")
                                  << ", timelineResolutionHz=" << (resHz > 0 ? std::to_string(resHz) : std::string("(default)"))
                                  << std::endl;
                    }
                }
            }
        } else {
            std::cerr << "[Config] No 'modules' object found in configuration; using defaults for all modules." << std::endl;
        }

        // Validate dependencies
        if (!pipeline->validateDependencies()) {
            std::cerr << "Error: Circular dependencies detected!" << std::endl;
            return 1;
        }

        // Show execution order
        std::cout << "\nExecution order:" << std::endl;
        for (const auto& moduleName : pipeline->getExecutionOrder()) {
            std::cout << "  - " << moduleName << std::endl;
        }

        // 3. Run analysis
        std::cout << "\nRunning analysis..." << std::endl;

        // Progress callback
        auto progressCallback = [](const std::string& module, float progress) {
            std::cout << "  [" << module << "] "
                     << static_cast<int>(progress * 100) << "%" << std::endl;
        };

        nlohmann::json analysisResult = pipeline->analyze(audioBuffer, progressCallback);

        // 4. Add processing time
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            endTime - startTime).count();
        analysisResult["analysisMetadata"]["processingTime"] = duration / 1000.0;

        // 5. Validate output
        if (!ave::core::JsonContract::validate(analysisResult)) {
            std::cerr << "Warning: Output validation failed!" << std::endl;
        }

        // 6. Save to file
        std::cout << "\nSaving results to: " << outputFile << std::endl;
        std::ofstream outFile(outputFile);
        outFile << analysisResult.dump(2);
        outFile.close();

        // 7. Print summary
        std::cout << "\n=== Analysis Complete ===" << std::endl;
        std::cout << "Processing time: " << duration / 1000.0 << " seconds" << std::endl;

        if (analysisResult.contains("tempo")) {
            std::cout << "BPM: " << analysisResult["tempo"]["bpm"]
                     << " (confidence: " << analysisResult["tempo"]["confidence"] << ")" << std::endl;
        }

        if (analysisResult.contains("tonality")) {
            std::cout << "Key: " << analysisResult["tonality"]["key"]
                     << " (confidence: " << analysisResult["tonality"]["confidence"] << ")" << std::endl;
        }

        if (analysisResult.contains("structure")) {
            std::cout << "Structure segments: " << analysisResult["structure"].size() << std::endl;
        }

        if (analysisResult.contains("cues")) {
            std::cout << "Cues detected: " << analysisResult["cues"].size() << std::endl;
        }

        std::cout << "\nOutput saved to: " << outputFile << std::endl;
        std::cout << "File size: " << analysisResult.dump().size() / 1024.0 << " KB" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}