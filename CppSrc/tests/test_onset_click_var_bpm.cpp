#include <iostream>
#include <vector>
#include <cmath>
#include <nlohmann/json.hpp>
#include "../include/core/AudioBuffer.h"
#include "../include/core/IAnalysisModule.h"
#include "../include/modules/OnsetModule.h"

using ave::core::AudioBuffer;
using ave::core::AnalysisContext;

static void synth_click_track(AudioBuffer& buf, double bpm, double clickAmp = 1.0, int clickLen = 192) {
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

bool test_onset_on_clicktrack_90bpm() {
    const size_t sr = 44100;
    const double bpm = 90.0; // 0.666.. s between onsets
    const double durSec = 12.0; // exactly 18 beats ideally

    AudioBuffer buf(1, static_cast<size_t>(sr * durSec), static_cast<float>(sr));
    synth_click_track(buf, bpm, 1.0, 192);

    auto onset = ave::modules::createRealOnsetModule();
    nlohmann::json cfg = {
        {"fftSize", 1024},
        {"hopSize", 512},
        {"windowType", "hann"},
        {"sensitivity", 1.0},
        {"peakMeanWindow", 16},
        {"peakThreshold", 0.01}
    };
    onset->initialize(cfg);

    AnalysisContext ctx; ctx.sampleRate = static_cast<float>(sr);
    nlohmann::json out = onset->process(buf, ctx);
    if (!onset->validateOutput(out)) {
        std::cerr << "Onset output validation failed" << std::endl;
        return false;
    }

    auto& onsets = out["onsets"];
    size_t count = onsets.size();
    std::cout << "90BPM detected onsets: " << count << std::endl;

    // Accept 17 or 18 due to end boundary
    if (!(count == 17 || count == 18)) {
        std::cerr << "Expected 17 or 18 onsets, got " << count << std::endl;
        return false;
    }

    // Alignment within 25 ms
    const double tol = 0.025;
    const double interval = 60.0 / bpm; // ~0.6667s

    std::vector<double> times; times.reserve(count);
    for (auto& o : onsets) times.push_back(o["t"].get<double>());

    size_t matches = 0;
    for (size_t i = 0; i < 18; ++i) {
        double ideal = i * interval;
        double best = 1e9;
        for (double t : times) best = std::min(best, std::abs(t - ideal));
        if (best <= tol) ++matches;
    }
    if (matches < 17) {
        std::cerr << "Alignment check failed at 90 BPM: matched " << matches << "/18 within 25ms" << std::endl;
        return false;
    }

    return true;
}
