//
// Created by jeanc on 8/27/2025.
//

#include <iostream>
#include <string>
#include <cmath>
#include "../include/pipeline/AnalysisPipeline.h"
#include "../include/pipeline/AudioLoader.h"
#include "../include/modules/BPMModule.h"

static bool run_bpm_pipeline_on_wav(const std::string& wavPath, double expectedBPM, double tol = 0.02) {
    try {
        auto audio = ave::pipeline::AudioLoader::loadWav(wavPath);

        auto pipeline = std::make_unique<ave::pipeline::AnalysisPipeline>();
        pipeline->registerModule(ave::modules::createRealBPMModule());
        pipeline->setModuleConfig("BPM", {
            {"minBPM", 60},
            {"maxBPM", 180},
            {"frameSize", 1024},
            {"hopSize", 512},
            {"acfWindowSec", 8.0},
            {"historySize", 10},
            {"octaveCorrection", true}
        });

        auto result = pipeline->analyze(audio);
        if (!result.contains("tempo")) {
            std::cerr << "No tempo in result for " << wavPath << std::endl;
            return false;
        }
        double bpm = result["tempo"]["bpm"].get<double>();
        double err = std::abs(bpm - expectedBPM) / expectedBPM;
        std::cout << "WAV: " << wavPath << " -> BPM " << bpm << " (err=" << err * 100 << "%)" << std::endl;
        return err <= tol;
    } catch (const std::exception& e) {
        std::cerr << "Exception in pipeline test for '" << wavPath << "': " << e.what() << std::endl;
        return false;
    }
}

bool test_pipeline_wav_120() {
    std::string wav = std::string("assets/test_120bpm.wav");
    return run_bpm_pipeline_on_wav(wav, 120.0);
}

bool test_pipeline_wav_130() {
    std::string wav = std::string("assets/test_130bpm.wav");
    return run_bpm_pipeline_on_wav(wav, 130.0);
}
