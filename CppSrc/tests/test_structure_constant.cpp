#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "../include/core/AudioBuffer.h"
#include "../include/core/IAnalysisModule.h"
#include "../include/modules/SpectralModule.h"
#include "../include/modules/StructureModule.h"

using ave::core::AudioBuffer;
using ave::core::AnalysisContext;

static AudioBuffer make_white_noise(float sr, float durSec, float amp = 0.3f) {
    size_t frames = static_cast<size_t>(durSec * sr);
    AudioBuffer buf(1, frames, sr);
    float* d = buf.getChannel(0);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t n = 0; n < frames; ++n) d[n] = amp * dist(rng);
    return buf;
}

bool test_structure_on_constant_noise() {
    const float sr = 44100.0f;
    const float dur = 15.0f;

    AudioBuffer buf = make_white_noise(sr, dur, 0.5f);

    // Spectral analysis
    auto spec = ave::modules::createRealSpectralModule();
    nlohmann::json scfg = {
        {"fftSize", 2048},
        {"hopSize", 512},
        {"windowType", "hann"}
    };
    spec->initialize(scfg);
    AnalysisContext ctx; ctx.sampleRate = sr;
    nlohmann::json specOut = spec->process(buf, ctx);

    // Structure analysis
    auto struc = ave::modules::createRealStructureModule();
    nlohmann::json cfg = {
        {"segmentMinLength", 8.0},
        {"noveltyKernelSize", 512},
        {"peakMeanWindow", 256},
        {"peakThreshold", 3.0},
        {"debug", false}
    };
    struc->initialize(cfg);

    AnalysisContext sctx; sctx.sampleRate = sr; sctx.moduleResults["Spectral"] = specOut;
    nlohmann::json out = struc->process(buf, sctx);
    if (!struc->validateOutput(out)) {
        std::cerr << "Structure output validation failed" << std::endl; return false; }

    if (!out.contains("segments")) { std::cerr << "Missing segments" << std::endl; return false; }

    auto& segs = out["segments"];
    size_t boundaries = (segs.size() > 0) ? (segs.size() - 1) : 0;
    std::cout << "Detected boundaries: " << boundaries << std::endl;

    // Expect none
    return boundaries == 0;
}
