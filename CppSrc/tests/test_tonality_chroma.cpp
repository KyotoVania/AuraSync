#include <iostream>
#include <vector>
#include "../include/core/AudioBuffer.h"
#include "../include/core/IAnalysisModule.h"
#include "../include/modules/TonalityModule.h"

using ave::core::AudioBuffer;
using ave::core::AnalysisContext;

static AudioBuffer make_sine(float freq, float sr, float durSec, float amp = 0.8f) {
    size_t frames = static_cast<size_t>(durSec * sr);
    AudioBuffer buf(1, frames, sr);
    float* d = buf.getChannel(0);
    const double twopi = 2.0 * M_PI;
    for (size_t n = 0; n < frames; ++n) {
        d[n] = amp * static_cast<float>(std::sin(twopi * freq * (static_cast<double>(n) / sr)));
    }
    return buf;
}

bool test_tonality_chroma_on_a440() {
    const float sr = 44100.0f;
    AudioBuffer buf = make_sine(440.0f, sr, 2.0f, 0.9f);

    auto ton = ave::modules::createRealTonalityModule();
    nlohmann::json cfg = {
        {"fftSize", 4096},
        {"hopSize", 2048},
        {"windowType", "hann"},
        {"referenceFreq", 440.0}
    };
    ton->initialize(cfg);

    AnalysisContext ctx; ctx.sampleRate = sr;
    nlohmann::json out = ton->process(buf, ctx);
    if (!ton->validateOutput(out)) {
        std::cerr << "Tonality output validation failed" << std::endl;
        return false;
    }

    if (!out.contains("chromaVector")) {
        std::cerr << "Missing chromaVector" << std::endl; return false;
    }

    auto chroma = out["chromaVector"].get<std::vector<double>>();
    if (chroma.size() != 12) { std::cerr << "Wrong chroma size" << std::endl; return false; }

    // Expect dominant at A = index 9 (C=0,...,A=9)
    size_t maxIdx = 0; double maxVal = chroma[0]; double sum = 0.0;
    for (size_t i = 0; i < 12; ++i) { sum += chroma[i]; if (chroma[i] > maxVal) { maxVal = chroma[i]; maxIdx = i; } }

    double dominance = (sum > 0.0) ? (maxVal / sum) : 0.0;
    std::cout << "A440 chroma peak index=" << maxIdx << ", peak share=" << dominance << std::endl;

    return maxIdx == 9 && dominance > 0.8;
}
