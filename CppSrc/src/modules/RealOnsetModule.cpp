#include "../../include/core/IAnalysisModule.h"
#include "../../include/core/AudioBuffer.h"
#include "../../include/modules/OnsetModule.h"
#include <nlohmann/json.hpp>
#include <fftw3.h>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace ave { namespace modules {

class RealOnsetModule : public core::IAnalysisModule {
public:
    std::string getName() const override { return "Onset"; }
    std::string getVersion() const override { return "1.0.0"; }
    bool isRealTime() const override { return true; }

    bool initialize(const nlohmann::json& config) override {
        if (config.contains("fftSize")) m_fftSize = std::max<size_t>(256, config["fftSize"].get<size_t>());
        if (config.contains("hopSize")) m_hopSize = std::max<size_t>(1, config["hopSize"].get<size_t>());
        if (m_hopSize > m_fftSize) m_hopSize = m_fftSize / 2; // clamp
        if (config.contains("windowType")) m_windowType = config["windowType"].get<std::string>();
        if (config.contains("sensitivity")) m_sensitivity = config["sensitivity"].get<double>();
        if (config.contains("peakMeanWindow")) m_peakMeanWindow = std::max<int>(1, config["peakMeanWindow"].get<int>());
        if (config.contains("peakThreshold")) m_peakThreshold = config["peakThreshold"].get<double>();
        if (config.contains("debug")) m_debug = config["debug"].get<bool>();
        return true;
    }

    void reset() override {}

    std::vector<std::string> getDependencies() const override { return {}; }

    nlohmann::json process(const core::AudioBuffer& audio, const core::AnalysisContext&) override {
        const float sr = audio.getSampleRate();
        const size_t N = m_fftSize;
        const size_t H = m_hopSize == 0 ? std::max<size_t>(1, N / 4) : m_hopSize;

        // Prepare mono signal
        std::vector<float> mono = audio.getMono();
        if (mono.empty()) {
            return makeEmptyResult();
        }

        // Window
        std::vector<float> winF;
        if (m_windowType == "hann" || m_windowType.empty()) winF = core::window::hann(N);
        else if (m_windowType == "hamming") winF = core::window::hamming(N);
        else if (m_windowType == "blackman") winF = core::window::blackman(N);
        else winF = core::window::hann(N);
        std::vector<double> window(winF.begin(), winF.end());

        // Number of frames (include last partial, zero-padded)
        size_t numFrames = (mono.size() + H - 1) / H;
        if (numFrames < 3) {
            return makeEmptyResult();
        }

        // FFTW setup
        double* in = (double*)fftw_malloc(sizeof(double) * N);
        if (!in) return makeEmptyResult();
        fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
        if (!out) { fftw_free(in); return makeEmptyResult(); }
        fftw_plan plan = fftw_plan_dft_r2c_1d(static_cast<int>(N), in, out, FFTW_ESTIMATE);

        std::vector<double> prevLogMag(N / 2 + 1, std::log(1e-10));
        std::vector<double> odf; odf.reserve(numFrames);

        for (size_t f = 0; f < numFrames; ++f) {
            const size_t start = f * H;
            for (size_t i = 0; i < N; ++i) {
                size_t idx = start + i;
                double s = (idx < mono.size()) ? static_cast<double>(mono[idx]) : 0.0;
                in[i] = s * window[i];
            }
            fftw_execute(plan);

            double flux = 0.0;
            for (size_t k = 0; k < (N / 2 + 1); ++k) {
                double re = out[k][0];
                double im = out[k][1];
                double mag = std::hypot(re, im);
                double logMag = std::log(mag + 1e-10);
                double d = logMag - prevLogMag[k];
                if (d > 0.0) flux += d; // half-wave rectified spectral flux (log magnitude)
                prevLogMag[k] = logMag;
            }
            odf.push_back(flux);
        }

        fftw_destroy_plan(plan);
        fftw_free(in);
        fftw_free(out);

        // Smooth ODF with small moving average to reduce ripples
        std::vector<double> odfSm(odf.size(), 0.0);
        int smoothRadius = 4; // window radius -> 9-point average
        if (!odf.empty()) {
            for (size_t t = 0; t < odf.size(); ++t) {
                int a = static_cast<int>(t) - smoothRadius;
                int b = static_cast<int>(t) + smoothRadius;
                a = std::max<int>(0, a);
                b = std::min<int>(static_cast<int>(odf.size()) - 1, b);
                double sum = 0.0; int cnt = 0;
                for (int i = a; i <= b; ++i) { sum += odf[static_cast<size_t>(i)]; ++cnt; }
                odfSm[t] = cnt ? (sum / cnt) : 0.0;
            }
        }

        // Peak picking: local maxima above adaptive threshold on smoothed ODF
        std::vector<size_t> peakIdx;
        if (odfSm.size() >= 3) {
            const int W = m_peakMeanWindow;
            const int minDist = std::max(1, W / 2);
            const int prePost = std::max(1, W / 2);
            for (size_t t = 1; t + 1 < odfSm.size(); ++t) {
                // Wider local maximum test over [t-prePost, t+prePost]
                int pmA = static_cast<int>(t) - prePost;
                int pmB = static_cast<int>(t) + prePost;
                pmA = std::max<int>(0, pmA);
                pmB = std::min<int>(static_cast<int>(odfSm.size()) - 1, pmB);
                bool isMax = true;
                for (int i = pmA; i <= pmB; ++i) {
                    if (odfSm[static_cast<size_t>(i)] > odfSm[t]) { isMax = false; break; }
                }
                if (!isMax) continue;
                int a = static_cast<int>(t) - W;
                int b = static_cast<int>(t) - 1; // causal mean: up to t-1
                a = std::max<int>(0, a);
                b = std::max<int>(a, b);
                double sum = 0.0; int cnt = 0;
                for (int i = a; i <= b; ++i) { sum += odfSm[static_cast<size_t>(i)]; ++cnt; }
                double mean = cnt ? (sum / cnt) : 0.0;
                if (mean < 1e-6) continue; // gate out near-silence regions
                double mult = (1.0 + (m_sensitivity * m_peakThreshold));
                double thresh = mean * mult;
                if (odfSm[t] >= thresh) {
                    // Enforce minimum distance between peaks
                    if (!peakIdx.empty()) {
                        if (static_cast<int>(t) - static_cast<int>(peakIdx.back()) < minDist) {
                            // Keep the stronger peak
                            if (odfSm[t] > odfSm[peakIdx.back()]) {
                                peakIdx.back() = t;
                            }
                            continue;
                        }
                    }
                    peakIdx.push_back(t);
                }
            }
        }

        // Build output (compensate timestamp by smoothing radius in frames)
        nlohmann::json onsets = nlohmann::json::array();
        for (size_t idx : peakIdx) {
            double tSec = static_cast<double>(idx + static_cast<size_t>(smoothRadius)) * static_cast<double>(H) / static_cast<double>(sr);
            // Use the smoothed ODF value as peak strength for robustness
            onsets.push_back({ {"t", tSec}, {"strength", odfSm[idx]} });
        }

        nlohmann::json result = {
            {"onsets", onsets},
            {"count", onsets.size()},
            {"sensitivity", m_sensitivity}
        };
        if (m_debug) {
            nlohmann::json odfArr = nlohmann::json::array();
            for (size_t i = 0; i < odfSm.size(); ++i) {
                double tSec = static_cast<double>(i) * static_cast<double>(H) / static_cast<double>(sr);
                odfArr.push_back({{"t", tSec}, {"v", odfSm[i]}});
            }
            result["debug"] = nlohmann::json{{"odf", odfArr}};
        }
        return result;
    }

    bool validateOutput(const nlohmann::json& output) const override {
        return output.contains("onsets") && output.contains("count");
    }

private:
    size_t m_fftSize = 1024;
    size_t m_hopSize = 512;
    std::string m_windowType = "hann";
    double m_sensitivity = 1.0;
    int m_peakMeanWindow = 8;      // W in frames
    double m_peakThreshold = 0.05; // delta
    bool m_debug = false;

    static nlohmann::json makeEmptyResult() {
        return nlohmann::json{
            {"onsets", nlohmann::json::array()},
            {"count", 0},
            {"sensitivity", 1.0}
        };
    }
};

std::unique_ptr<core::IAnalysisModule> createRealOnsetModule() {
    return std::make_unique<RealOnsetModule>();
}

} } // namespace ave::modules