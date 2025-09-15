#include "../../include/core/IAnalysisModule.h"
#include "../../include/core/AudioBuffer.h"
#include "../../include/modules/SpectralModule.h"
#include <nlohmann/json.hpp>
#include <fftw3.h>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>

namespace ave {
namespace modules {

class RealSpectralModule : public core::IAnalysisModule {
public:
    std::string getName() const override { return "Spectral"; }
    std::string getVersion() const override { return "1.0.0"; }
    bool isRealTime() const override { return true; }

    bool initialize(const nlohmann::json& config) override {
        if (config.contains("fftSize")) m_fftSize = std::max<size_t>(32, config["fftSize"].get<size_t>());
        if (config.contains("hopSize")) m_hopSize = std::max<size_t>(1, config["hopSize"].get<size_t>());
        if (m_hopSize > m_fftSize) m_hopSize = m_fftSize / 2; // clamp
        if (config.contains("windowType")) m_windowType = config["windowType"].get<std::string>();
        if (config.contains("bandDefinitions")) {
            m_bandDefs = config["bandDefinitions"];
        } else {
            // Default 5 bands
            m_bandDefs = {
                {"low", {0.0, 250.0}},
                {"lowMid", {250.0, 500.0}},
                {"mid", {500.0, 2000.0}},
                {"highMid", {2000.0, 4000.0}},
                {"high", {4000.0, 22050.0}}
            };
        }
        return true;
    }

    void reset() override {
        // No persistent FFTW plan kept between process() calls in this simple implementation
    }

    nlohmann::json process(const core::AudioBuffer& audio, const core::AnalysisContext& context) override {
        const float sampleRate = audio.getSampleRate();
        const size_t N = m_fftSize;
        const size_t H = m_hopSize == 0 ? std::max<size_t>(1, N / 4) : m_hopSize;
        if (N < 32) {
            return makeEmptyResult(sampleRate, N, H);
        }

        // Prepare mono signal
        std::vector<float> mono = audio.getMono();
        if (mono.empty()) {
            return makeEmptyResult(sampleRate, N, H);
        }

        // Precompute window
        std::vector<float> winF;
        if (m_windowType == "hann" || m_windowType.empty()) {
            winF = core::window::hann(N);
        } else if (m_windowType == "hamming") {
            winF = core::window::hamming(N);
        } else if (m_windowType == "blackman") {
            winF = core::window::blackman(N);
        } else {
            winF = core::window::hann(N);
        }
        std::vector<double> window(winF.begin(), winF.end());

        // Determine number of frames with zero-padding for last partial frame
        const size_t numFrames = (mono.size() + H - 1) / H; // process until start < mono.size()

        // FFTW setup (plan once, execute per frame)
        double* in = (double*)fftw_malloc(sizeof(double) * N);
        if (!in) return makeEmptyResult(sampleRate, N, H);
        fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
        if (!out) {
            fftw_free(in);
            return makeEmptyResult(sampleRate, N, H);
        }
        fftw_plan plan = fftw_plan_dft_r2c_1d(static_cast<int>(N), in, out, FFTW_ESTIMATE);

        // Prepare band names and bin mapping
        std::vector<std::string> bandNames;
        std::vector<std::pair<double,double>> bandRanges;
        bandNames.reserve(m_bandDefs.size());
        bandRanges.reserve(m_bandDefs.size());
        const double nyquist = sampleRate / 2.0;
        for (auto it = m_bandDefs.begin(); it != m_bandDefs.end(); ++it) {
            const std::string name = it.key();
            const auto arr = it.value();
            double lo = 0.0, hi = 0.0;
            if (arr.is_array() && arr.size() >= 2) {
                lo = arr[0].get<double>();
                hi = arr[1].get<double>();
            }
            if (lo < 0.0) lo = 0.0;
            if (hi > nyquist) hi = nyquist;
            if (hi < lo) std::swap(hi, lo);
            bandNames.push_back(name);
            bandRanges.emplace_back(lo, hi);
        }

        const size_t numBins = N / 2 + 1;
        std::vector<int> binToBand(numBins, -1);
        for (size_t k = 0; k < numBins; ++k) {
            double fk = (static_cast<double>(k) * sampleRate) / static_cast<double>(N);
            for (size_t b = 0; b < bandRanges.size(); ++b) {
                if (fk >= bandRanges[b].first && fk <= bandRanges[b].second) {
                    binToBand[k] = static_cast<int>(b);
                    break;
                }
            }
        }

        // Output structure
        nlohmann::json bands = nlohmann::json::object();
        for (const auto& name : bandNames) bands[name] = nlohmann::json::array();

        // STFT loop
        for (size_t f = 0; f < numFrames; ++f) {
            const size_t start = f * H;
            // Fill input with windowed samples or zeros beyond length
            for (size_t i = 0; i < N; ++i) {
                size_t idx = start + i;
                double s = (idx < mono.size()) ? static_cast<double>(mono[idx]) : 0.0;
                in[i] = s * window[i];
            }

            // Execute FFT
            fftw_execute(plan);

            // Aggregate power per band
            std::vector<double> bandEnergy(bandNames.size(), 0.0);
            for (size_t k = 0; k < numBins; ++k) {
                double re = out[k][0];
                double im = out[k][1];
                double p = re * re + im * im; // power spectrum
                int b = binToBand[k];
                if (b >= 0) bandEnergy[static_cast<size_t>(b)] += p;
            }

            // Timestamp (frame start in seconds as per spec)
            double t = static_cast<double>(f) * static_cast<double>(H) / static_cast<double>(sampleRate);

            // Append to output arrays
            for (size_t b = 0; b < bandNames.size(); ++b) {
                bands[bandNames[b]].push_back({ {"t", t}, {"v", static_cast<float>(bandEnergy[b])} });
            }
        }

        // Cleanup
        fftw_destroy_plan(plan);
        fftw_free(in);
        fftw_free(out);

        const double frameRate = static_cast<double>(sampleRate) / static_cast<double>(H);
        nlohmann::json result = {
            {"bands", bands},
            {"fftSize", N},
            {"hopSize", H},
            {"frameRate", frameRate}
        };
        return result;
    }

    bool validateOutput(const nlohmann::json& output) const override {
        return output.contains("bands") && output.contains("frameRate");
    }

private:
    size_t m_fftSize = 2048;
    size_t m_hopSize = 512;
    std::string m_windowType = "hann";
    nlohmann::json m_bandDefs = nlohmann::json::object();

    static nlohmann::json makeEmptyResult(float sampleRate, size_t N, size_t H) {
        if (H == 0) H = std::max<size_t>(1, N / 4);
        return {
            {"bands", nlohmann::json::object()},
            {"fftSize", N},
            {"hopSize", H},
            {"frameRate", sampleRate / static_cast<float>(H)}
        };
    }
};

std::unique_ptr<core::IAnalysisModule> createRealSpectralModule() {
    return std::make_unique<RealSpectralModule>();
}

} // namespace modules
} // namespace ave
