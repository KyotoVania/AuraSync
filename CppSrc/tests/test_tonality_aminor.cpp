#include <iostream>
#include <vector>
#include "../include/core/AudioBuffer.h"
#include "../include/core/IAnalysisModule.h"
#include "../include/modules/TonalityModule.h"

using ave::core::AudioBuffer;
using ave::core::AnalysisContext;

static AudioBuffer make_sine(float freq, float sr, float durSec, float amp = 0.6f) {
    size_t frames = static_cast<size_t>(durSec * sr);
    AudioBuffer buf(1, frames, sr);
    float* d = buf.getChannel(0);
    const double twopi = 2.0 * M_PI;
    for (size_t n = 0; n < frames; ++n) {
        d[n] = amp * static_cast<float>(std::sin(twopi * freq * (static_cast<double>(n) / sr)));
    }
    return buf;
}

bool test_tonality_on_aminor_chord() {
    const float sr = 44100.0f;
    const float dur = 2.0f;
    // A minor chord: A, C, E (use A4, C5, E5)
    std::vector<float> freqs = {440.0f, 523.25f, 659.25f};
    AudioBuffer buf(1, static_cast<size_t>(dur * sr), sr);
    float* d = buf.getChannel(0);
    for (auto f : freqs) {
        AudioBuffer s = make_sine(f, sr, dur, 0.25f);
        const float* sd = s.getChannel(0);
        for (size_t n = 0; n < buf.getFrameCount(); ++n) d[n] += sd[n];
    }

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
        std::cerr << "Tonality output validation failed" << std::endl; return false; }

    std::string key = out["key"].get<std::string>();
    std::string mode = out["mode"].get<std::string>();
    double conf = out["confidence"].get<double>();
    std::cout << "Detected key=" << key << " mode=" << mode << " conf=" << conf << std::endl;

    return (key == "A" && mode == "minor" && conf > 0.7);
}
