#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>
#include "../include/pipeline/AnalysisPipeline.h"
#include "../include/core/AudioBuffer.h"
#include "../include/core/JsonContract.h"
#include "../include/modules/BPMModule.h"

// Forward declarations for fake module factories
namespace ave::modules {
    std::unique_ptr<ave::core::IAnalysisModule> createFakeOnsetModule();
    std::unique_ptr<ave::core::IAnalysisModule> createFakeStructureModule();
    std::unique_ptr<ave::core::IAnalysisModule> createFakeTonalityModule();
    std::unique_ptr<ave::core::IAnalysisModule> createFakeSpectralModule();
    std::unique_ptr<ave::core::IAnalysisModule> createFakeCueModule();
}

/**
 * Fake audio loader for testing
 * In real implementation, will use libsndfile
 */
ave::core::AudioBuffer loadAudioFile(const std::string& path) {
    std::cout << "Loading audio file: " << path << std::endl;

    // Create fake audio buffer for testing
    size_t sampleRate = 44100;
    size_t durationSec = 180; // 3 minutes
    size_t channels = 2;
    size_t frames = sampleRate * durationSec;

    ave::core::AudioBuffer buffer(channels, frames, sampleRate);

    // Fill with fake audio data (silence for now)
    for (size_t ch = 0; ch < channels; ++ch) {
        float* channelData = buffer.getChannel(ch);
        for (size_t i = 0; i < frames; ++i) {
            channelData[i] = 0.0f;
        }
    }

    std::cout << "Loaded " << durationSec << " seconds of audio ("
              << channels << " channels, " << sampleRate << " Hz)" << std::endl;

    return buffer;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Audio Visual Engine - Analysis Pipeline ===" << std::endl;
    std::cout << "Version: 1.0.0-prototype" << std::endl << std::endl;

    // Parse arguments
    std::string inputFile = argc > 1 ? argv[1] : "test.wav";
    std::string outputFile = argc > 2 ? argv[2] : "analysis.json";

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
        pipeline->registerModule(ave::modules::createFakeOnsetModule());
        pipeline->registerModule(ave::modules::createFakeStructureModule());
        pipeline->registerModule(ave::modules::createFakeTonalityModule());
        pipeline->registerModule(ave::modules::createFakeSpectralModule());
        pipeline->registerModule(ave::modules::createFakeCueModule());

        // Configure modules
        pipeline->setModuleConfig("BPM", {
            {"minBPM", 60},
            {"maxBPM", 180},
            {"frameSize", 1024},
            {"hopSize", 512}
        });

        pipeline->setModuleConfig("Onset", {
            {"sensitivity", 0.5}
        });

        pipeline->setModuleConfig("Structure", {
            {"segmentMinLength", 8.0}
        });

        pipeline->setModuleConfig("Spectral", {
            {"fftSize", 2048},
            {"hopSize", 512}
        });

        pipeline->setModuleConfig("Cue", {
            {"anticipationTime", 1.5}
        });

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