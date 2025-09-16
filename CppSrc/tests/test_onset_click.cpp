#include <iostream>
#include <vector>
#include <cmath>
#include <nlohmann/json.hpp>
#include "../include/core/AudioBuffer.h"
#include "../include/core/IAnalysisModule.h"
#include "../include/modules/OnsetModule.h"

using ave::core::AudioBuffer;
using ave::core::AnalysisContext;

static void synth_click_track(AudioBuffer& buf, double bpm, double clickAmp = 1.0, int clickLen = 128) {
    double sr = buf.getSampleRate();
    double interval = 60.0 / bpm;
    size_t hop = static_cast<size_t>(interval * sr);
    for (size_t ch = 0; ch < buf.getChannelCount(); ++ch) {
        float* data = buf.getChannel(ch);
        for (size_t i = 0; i < buf.getFrameCount(); ++i) data[i] = 0.0f;
        for (size_t i = 0; i < buf.getFrameCount(); i += hop) {
            for (int k = 0; k < clickLen && (i + k) < buf.getFrameCount(); ++k) {
                data[i + k] += static_cast<float>(clickAmp * std::exp(-0.03 * k));
            }
        }
    }
}

bool test_onset_on_clicktrack() {
    const size_t sr = 44100;
    const double bpm = 120.0;
    const double durSec = 10.0;

    AudioBuffer buf(1, static_cast<size_t>(sr * durSec), static_cast<float>(sr));
    synth_click_track(buf, bpm, 1.0, 256);

    auto onset = ave::modules::createRealOnsetModule();
    nlohmann::json cfg = {
        {"fftSize", 1024},
        {"hopSize", 512},
        {"windowType", "hann"},
        {"sensitivity", 1.0},
        {"peakMeanWindow", 16},
        {"peakThreshold", 0.01},
        {"debug", true}
    };
    onset->initialize(cfg);

    AnalysisContext ctx; ctx.sampleRate = static_cast<float>(sr);
    nlohmann::json out = onset->process(buf, ctx);
    if (!onset->validateOutput(out)) {
        std::cerr << "Onset output validation failed" << std::endl;
        return false;
    }

    // Debug: print ODF stats if available
    if (out.contains("debug") && out["debug"].contains("odf")) {
        double mean = 0.0, mx = -1e9; size_t c = 0; double mnn = 1e9;
        for (auto& pt : out["debug"]["odf"]) {
            double v = pt["v"].get<double>();
            mean += v; ++c; mx = std::max(mx, v); mnn = std::min(mnn, v);
        }
        if (c) mean /= c; else mean = 0.0;
        std::cout << "ODF stats -> mean: " << mean << ", min: " << mnn << ", max: " << mx << std::endl;
    }

    auto& onsets = out["onsets"];
    size_t count = onsets.size();
    std::cout << "Detected onsets: " << count << std::endl;

    if (!(count == 19 || count == 20)) {
        std::cerr << "Expected 19 or 20 onsets, got " << count << std::endl;
        return false;
    }

    // Check timing alignment within +/-25 ms
    const double tol = 0.025; // 25 ms
    const double interval = 0.5; // 120 BPM -> 0.5s

    // Build vector of detected times
    std::vector<double> times;
    times.reserve(count);
    for (auto& o : onsets) times.push_back(o["t"].get<double>());

    size_t matches = 0;
    for (size_t i = 0; i < 20; ++i) {
        double ideal = i * interval;
        // Find nearest detected time
        double best = 1e9;
        for (double t : times) best = std::min(best, std::abs(t - ideal));
        if (best <= tol) ++matches;
    }

    if (matches < 19) {
        std::cerr << "Alignment check failed: matched " << matches << "/20 within 25ms" << std::endl;
        std::cerr << "First detected times:";
        for (size_t i = 0; i < std::min<size_t>(times.size(), 10); ++i) {
            std::cerr << " " << times[i];
        }
        std::cerr << std::endl;
        return false;
    }

    return true;
}
