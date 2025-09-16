#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <nlohmann/json.hpp>
#include "../include/core/AudioBuffer.h"
#include "../include/core/IAnalysisModule.h"
#include "../include/modules/OnsetModule.h"

using ave::core::AudioBuffer;
using ave::core::AnalysisContext;

static void synth_noise_bursts(AudioBuffer& buf, const std::vector<double>& times, double burstDurSec = 0.01, double amp = 0.9) {
    const size_t sr = static_cast<size_t>(buf.getSampleRate());
    float* d = buf.getChannel(0);
    for (size_t i = 0; i < buf.getFrameCount(); ++i) d[i] = 0.0f;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    size_t burstLen = static_cast<size_t>(burstDurSec * sr);
    for (double t : times) {
        size_t start = static_cast<size_t>(t * sr);
        for (size_t k = 0; k < burstLen && (start + k) < buf.getFrameCount(); ++k) {
            d[start + k] += static_cast<float>(amp * dist(rng));
        }
    }
}

bool test_onset_on_noise_bursts() {
    const float sr = 44100.0f;
    const double durSec = 8.0;
    const std::vector<double> burstTimes = {1.0, 2.5, 4.0, 6.0};

    AudioBuffer buf(1, static_cast<size_t>(sr * durSec), sr);
    synth_noise_bursts(buf, burstTimes, 0.012, 0.9);

    auto onset = ave::modules::createRealOnsetModule();
    nlohmann::json cfg = {
        {"fftSize", 1024},
        {"hopSize", 512},
        {"windowType", "hann"},
        {"sensitivity", 1.0},
        {"peakMeanWindow", 12},
        {"peakThreshold", 0.05}
    };
    onset->initialize(cfg);

    AnalysisContext ctx; ctx.sampleRate = sr;
    nlohmann::json out = onset->process(buf, ctx);

    auto& onsets = out["onsets"];
    size_t count = onsets.size();
    std::cout << "Noise bursts detected onsets: " << count << std::endl;

    if (count < burstTimes.size()) {
        std::cerr << "Expected at least " << burstTimes.size() << " onsets, got " << count << std::endl;
        return false;
    }

    // Check alignment within 30 ms
    const double tol = 0.03;
    size_t matches = 0;
    for (double ideal : burstTimes) {
        double best = 1e9;
        for (auto& o : onsets) {
            double t = o["t"].get<double>();
            best = std::min(best, std::abs(t - ideal));
        }
        if (best <= tol) ++matches;
    }

    if (matches < burstTimes.size()) {
        std::cerr << "Alignment failed for noise bursts: matched " << matches << "/" << burstTimes.size() << std::endl;
        return false;
    }

    return true;
}
