#include <iostream>
#include <vector>
#include <cmath>
#include <nlohmann/json.hpp>
#include "../include/core/AudioBuffer.h"
#include "../include/core/IAnalysisModule.h"
#include "../include/modules/OnsetModule.h"

using ave::core::AudioBuffer;
using ave::core::AnalysisContext;

static void synth_accented_clicks(AudioBuffer& buf, double bpm, double strongAmp = 1.0, double weakAmp = 0.3, int clickLen = 192) {
    double sr = buf.getSampleRate();
    double interval = 60.0 / bpm;
    size_t hop = static_cast<size_t>(interval * sr);
    float* data = buf.getChannel(0);
    for (size_t i = 0; i < buf.getFrameCount(); ++i) data[i] = 0.0f;
    size_t idx = 0;
    int beat = 0;
    while (idx < buf.getFrameCount()) {
        double amp = (beat % 2 == 0) ? strongAmp : weakAmp;
        for (int k = 0; k < clickLen && (idx + k) < buf.getFrameCount(); ++k) {
            data[idx + k] += static_cast<float>(amp * std::exp(-0.03 * k));
        }
        idx += hop;
        ++beat;
    }
}

bool test_onset_on_accented_clicks() {
    const size_t sr = 44100;
    const double bpm = 100.0; // 0.6s between onsets
    const double durSec = 12.0; // 20 beats ideally

    AudioBuffer buf(1, static_cast<size_t>(sr * durSec), static_cast<float>(sr));
    synth_accented_clicks(buf, bpm, 1.0, 0.3, 192);

    auto onset = ave::modules::createRealOnsetModule();
    nlohmann::json cfg = {
        {"fftSize", 1024},
        {"hopSize", 512},
        {"windowType", "hann"},
        {"sensitivity", 1.0},
        {"peakMeanWindow", 16},
        {"peakThreshold", 0.02}
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
    std::cout << "Accented clicks detected onsets: " << count << std::endl;

    if (count < 18) {
        std::cerr << "Expected around 20 onsets, got too few: " << count << std::endl;
        return false;
    }

    // Build detected times and strengths
    std::vector<std::pair<double,double>> det; det.reserve(count);
    for (auto& o : onsets) det.emplace_back(o["t"].get<double>(), o["strength"].get<double>());

    const double interval = 60.0 / bpm; // 0.6s
    const double tol = 0.03; // 30 ms tolerance

    std::vector<double> strongS, weakS;
    for (int i = 0; i < 20; ++i) {
        double ideal = i * interval;
        // find nearest detection
        double best = 1e9; size_t bestIdx = (size_t)-1;
        for (size_t j = 0; j < det.size(); ++j) {
            double d = std::abs(det[j].first - ideal);
            if (d < best) { best = d; bestIdx = j; }
        }
        if (best <= tol && bestIdx != (size_t)-1) {
            if ((i % 2) == 0) strongS.push_back(det[bestIdx].second);
            else weakS.push_back(det[bestIdx].second);
        }
    }

    double meanStrong = 0.0, meanWeak = 0.0;
    for (double v : strongS) meanStrong += v; if (!strongS.empty()) meanStrong /= strongS.size();
    for (double v : weakS) meanWeak += v; if (!weakS.empty()) meanWeak /= weakS.size();

    std::cout << "Accented strength means -> strong: " << meanStrong << ", weak: " << meanWeak << std::endl;

    if (strongS.size() < 6 || weakS.size() < 6) {
        std::cerr << "Not enough matched onsets for strength comparison" << std::endl;
        return false;
    }

    // Accents should be stronger (log-magnitude flux compresses dynamic range)
    return meanStrong > 1.04 * meanWeak;
}
