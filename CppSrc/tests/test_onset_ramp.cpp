#include <iostream>
#include <nlohmann/json.hpp>
#include "../include/core/AudioBuffer.h"
#include "../include/core/IAnalysisModule.h"
#include "../include/modules/OnsetModule.h"

using ave::core::AudioBuffer;
using ave::core::AnalysisContext;

static AudioBuffer make_ramp_sine(float sr, float durSec, float freq) {
    size_t frames = static_cast<size_t>(sr * durSec);
    AudioBuffer buf(1, frames, sr);
    float* d = buf.getChannel(0);
    const double twopi = 2.0 * M_PI;
    for (size_t n = 0; n < frames; ++n) {
        double t = static_cast<double>(n) / sr;
        double amp = static_cast<double>(n) / std::max<size_t>(1, frames - 1); // 0..1 linear ramp
        d[n] = static_cast<float>(amp * std::sin(twopi * freq * t));
    }
    return buf;
}

bool test_onset_on_ramp() {
    const float sr = 44100.0f;
    AudioBuffer buf = make_ramp_sine(sr, 2.0f, 440.0f);

    auto onset = ave::modules::createRealOnsetModule();
    nlohmann::json cfg = {
        {"fftSize", 1024},
        {"hopSize", 512},
        {"windowType", "hann"},
        {"sensitivity", 1.0},
        {"peakMeanWindow", 16},
        {"peakThreshold", 0.1}
    };
    onset->initialize(cfg);

    AnalysisContext ctx; ctx.sampleRate = sr;
    nlohmann::json out = onset->process(buf, ctx);

    size_t count = out["onsets"].size();
    std::cout << "Ramp onsets count: " << count << std::endl;
    return count == 0;
}
