//
// Created by jeanc on 8/27/2025.
//

#include <iostream>
#include <cmath>
#include <nlohmann/json.hpp>
#include "../include/core/AudioBuffer.h"
#include "../include/core/IAnalysisModule.h"
#include "../include/modules/BPMModule.h"

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
