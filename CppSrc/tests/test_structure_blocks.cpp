#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "../include/core/AudioBuffer.h"
#include "../include/core/IAnalysisModule.h"
#include "../include/modules/SpectralModule.h"
#include "../include/modules/StructureModule.h"

using ave::core::AudioBuffer;
using ave::core::AnalysisContext;

static AudioBuffer make_sine_block(float freq, float sr, float durSec, float amp = 0.6f) {
    size_t frames = static_cast<size_t>(durSec * sr);
    AudioBuffer buf(1, frames, sr);
    float* d = buf.getChannel(0);
    const double twopi = 2.0 * M_PI;
    for (size_t n = 0; n < frames; ++n) {
        d[n] = amp * static_cast<float>(std::sin(twopi * freq * (static_cast<double>(n) / sr)));
    }
    return buf;
}

bool test_structure_on_block_change() {
    const float sr = 44100.0f;
    const float dur = 15.0f;

    // 5s @1kHz + 5s @8kHz + 5s @1kHz
    AudioBuffer all(1, static_cast<size_t>(dur * sr), sr);
    float* d = all.getChannel(0);
    // Clear
    for (size_t i = 0; i < all.getFrameCount(); ++i) d[i] = 0.0f;

    auto b1 = make_sine_block(1000.0f, sr, 5.0f, 0.7f);
    auto b2 = make_sine_block(8000.0f, sr, 5.0f, 0.7f);
    auto b3 = make_sine_block(1000.0f, sr, 5.0f, 0.7f);

    // Mix blocks sequentially
    size_t f5 = static_cast<size_t>(5.0f * sr);
    const float* s1 = b1.getChannel(0);
    const float* s2 = b2.getChannel(0);
    const float* s3 = b3.getChannel(0);
    for (size_t n = 0; n < f5; ++n) d[n] = s1[n];
    for (size_t n = 0; n < f5; ++n) d[f5 + n] = s2[n];
    for (size_t n = 0; n < f5; ++n) d[2 * f5 + n] = s3[n];

    // Spectral analysis
    auto spec = ave::modules::createRealSpectralModule();
    nlohmann::json scfg = {
        {"fftSize", 2048},
        {"hopSize", 512},
        {"windowType", "hann"}
    };
    spec->initialize(scfg);
    AnalysisContext ctx; ctx.sampleRate = sr;
    nlohmann::json specOut = spec->process(all, ctx);

    // Structure analysis
    auto struc = ave::modules::createRealStructureModule();
    nlohmann::json cfg = {
        {"segmentMinLength", 4.0},      // seconds
        {"noveltyKernelSize", 128},      // frames
        {"peakMeanWindow", 64},          // frames
        {"peakThreshold", 0.25},         // relative to mean
        {"debug", false}
    };
    struc->initialize(cfg);

    // Provide spectral result to context
    AnalysisContext sctx; sctx.sampleRate = sr; sctx.moduleResults["Spectral"] = specOut;
    nlohmann::json out = struc->process(all, sctx);
    if (!struc->validateOutput(out)) {
        std::cerr << "Structure output validation failed" << std::endl; return false; }

    if (!out.contains("segments")) { std::cerr << "Missing segments" << std::endl; return false; }

    auto& segs = out["segments"];
    if (segs.size() < 2) {
        std::cerr << "Not enough segments: " << segs.size() << std::endl; return false; }

    // Boundaries are segment end times except the last one which is duration
    std::vector<double> boundaries;
    for (size_t i = 0; i + 1 < segs.size(); ++i) {
        boundaries.push_back(segs[i]["end"].get<double>());
    }

    // Expect two boundaries near 5.0s and 10.0s
    size_t count = boundaries.size();
    std::cout << "Detected boundaries: " << count << std::endl;
    for (double b : boundaries) std::cout << "  b@" << b;
    std::cout << std::endl;

    if (count == 0) return false;

    // Check proximity within tolerance
    auto nearest = [&](double x){ double best = 1e9; for (double b : boundaries) best = std::min(best, std::abs(b - x)); return best; };
    double tol = 0.5; // 500 ms tolerance
    bool ok = (nearest(5.0) <= tol) && (nearest(10.0) <= tol);

    return ok && (count >= 2);
}
