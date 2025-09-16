#include <iostream>
#include <nlohmann/json.hpp>
#include "../include/core/AudioBuffer.h"
#include "../include/core/IAnalysisModule.h"
#include "../include/modules/OnsetModule.h"

using ave::core::AudioBuffer;
using ave::core::AnalysisContext;

bool test_onset_on_silence() {
    const size_t sr = 44100;
    const double durSec = 2.0;

    AudioBuffer buf(1, static_cast<size_t>(sr * durSec), static_cast<float>(sr));
    // Already zeroed buffer (silence)

    auto onset = ave::modules::createRealOnsetModule();
    nlohmann::json cfg = {
        {"fftSize", 1024},
        {"hopSize", 512},
        {"windowType", "hann"},
        {"sensitivity", 1.0},
        {"peakMeanWindow", 8},
        {"peakThreshold", 0.05}
    };
    onset->initialize(cfg);

    AnalysisContext ctx; ctx.sampleRate = static_cast<float>(sr);
    nlohmann::json out = onset->process(buf, ctx);

    auto& onsets = out["onsets"];
    size_t count = onsets.size();
    std::cout << "Silence onsets count: " << count << std::endl;
    return count == 0;
}
