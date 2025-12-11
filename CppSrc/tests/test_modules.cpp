//
// Created by jeanc on 8/27/2025.
//

#include <iostream>
#include <cmath>
#include <random>
#include <map>
#include <nlohmann/json.hpp>
#include "../include/core/AudioBuffer.h"
#include "../include/core/IAnalysisModule.h"
#include "../include/modules/BPMModule.h"
#include "../include/modules/SpectralModule.h"

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
                // short decaying click
                data[i + k] += static_cast<float>(clickAmp * std::exp(-0.03 * k));
            }
        }
    }
}

bool test_bpm_on_clicktrack() {
    const size_t sr = 44100;
    const double targetBPM = 120.0;
    const double tol = 0.02; // 2%
    const double durSec = 10.0;

    AudioBuffer buf(2, static_cast<size_t>(sr * durSec), static_cast<float>(sr));
    synth_click_track(buf, targetBPM);

    auto bpmModule = ave::modules::createRealBPMModule();
    nlohmann::json cfg = {
        {"minBPM", 60}, {"maxBPM", 180}, {"frameSize", 1024}, {"hopSize", 512}
    };
    if (!bpmModule->initialize(cfg)) {
        std::cerr << "Failed to initialize BPM module" << std::endl;
        return false;
    }

    AnalysisContext ctx; ctx.sampleRate = static_cast<float>(sr);
    nlohmann::json out = bpmModule->process(buf, ctx);

    if (!bpmModule->validateOutput(out)) {
        std::cerr << "BPM output validation failed" << std::endl;
        return false;
    }

    double bpm = out["bpm"].get<double>();
    double err = std::abs(bpm - targetBPM) / targetBPM;
    std::cout << "Estimated BPM: " << bpm << " (err=" << err * 100 << "%)" << std::endl;
    return err <= tol;
}

static AudioBuffer make_sine(float freq, float sr, float durSec, float amp = 0.5f) {
    size_t frames = static_cast<size_t>(durSec * sr);
    AudioBuffer buf(1, frames, sr);
    float* d = buf.getChannel(0);
    const double twopi = 2.0 * M_PI;
    for (size_t n = 0; n < frames; ++n) {
        d[n] = amp * static_cast<float>(std::sin(twopi * freq * (static_cast<double>(n) / sr)));
    }
    return buf;
}

static AudioBuffer make_white_noise(float sr, float durSec, float amp = 0.3f) {
    size_t frames = static_cast<size_t>(durSec * sr);
    AudioBuffer buf(1, frames, sr);
    float* d = buf.getChannel(0);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t n = 0; n < frames; ++n) {
        d[n] = amp * dist(rng);
    }
    return buf;
}

bool test_spectral_on_sine_1000hz() {
    const float sr = 44100.0f;
    AudioBuffer buf = make_sine(1000.0f, sr, 2.0f, 0.6f);

    auto spec = ave::modules::createRealSpectralModule();
    nlohmann::json cfg = {
        {"fftSize", 2048},
        {"hopSize", 512},
        {"windowType", "hann"}
    };
    spec->initialize(cfg);

    AnalysisContext ctx; ctx.sampleRate = sr;
    nlohmann::json out = spec->process(buf, ctx);
    if (!spec->validateOutput(out)) {
        std::cerr << "Spectral output validation failed" << std::endl;
        return false;
    }

    auto avgBand = [&](const std::string& name){
        double s = 0.0; size_t c = 0;
        if (!out.contains("bands") || !out["bands"].contains(name)) return 0.0;
        for (auto& pt : out["bands"][name]) { s += pt["v"].get<double>(); ++c; }
        return c ? s / c : 0.0;
    };

    double low = avgBand("low");
    double lowMid = avgBand("lowMid");
    double mid = avgBand("mid");
    double highMid = avgBand("highMid");
    double high = avgBand("high");
    double total = low + lowMid + mid + highMid + high + 1e-12;
    double ratio = mid / total;

    std::cout << "Sine1000Hz band averages -> low:" << low << ", lowMid:" << lowMid
              << ", mid:" << mid << ", highMid:" << highMid << ", high:" << high
              << ", mid/total=" << ratio << std::endl;

    return ratio > 0.8; // mid band must dominate
}

bool test_spectral_on_white_noise() {
    const float sr = 44100.0f;
    AudioBuffer buf = make_white_noise(sr, 2.0f, 0.6f);

    auto spec = ave::modules::createRealSpectralModule();
    nlohmann::json cfg = {
        {"fftSize", 2048},
        {"hopSize", 512},
        {"windowType", "hann"}
    };
    spec->initialize(cfg);

    AnalysisContext ctx; ctx.sampleRate = sr;
    nlohmann::json out = spec->process(buf, ctx);
    if (!spec->validateOutput(out)) {
        std::cerr << "Spectral output validation failed" << std::endl;
        return false;
    }

    auto avgBand = [&](const std::string& name){
        double s = 0.0; size_t c = 0;
        if (!out.contains("bands") || !out["bands"].contains(name)) return 0.0;
        for (auto& pt : out["bands"][name]) { s += pt["v"].get<double>(); ++c; }
        return c ? s / c : 0.0;
    };

    double low = avgBand("low");
    double mid = avgBand("mid");

    double ratio = (low > 0.0) ? (mid / low) : 0.0;
    std::cout << "WhiteNoise band avg ratio mid/low = " << ratio << std::endl;

    return (ratio > 0.8 && ratio < 1.25);
}