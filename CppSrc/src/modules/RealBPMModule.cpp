#include "../../include/modules/BPMModule.h"
#include "../../include/core/IAnalysisModule.h"
#include "../../include/core/AudioBuffer.h"
#include <nlohmann/json.hpp>
#include <fftw3.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace ave::modules {

class RealBPMModule : public core::IAnalysisModule {
public:
    std::string getName() const override { return "BPM"; }
    std::string getVersion() const override { return "1.1.0"; }

    bool initialize(const nlohmann::json& config) override {
        if (config.contains("minBPM")) m_cfg.minBPM = config["minBPM"].get<float>();
        if (config.contains("maxBPM")) m_cfg.maxBPM = config["maxBPM"].get<float>();
        if (config.contains("frameSize")) m_cfg.frameSize = config["frameSize"].get<size_t>();
        if (config.contains("hopSize")) m_cfg.hopSize = config["hopSize"].get<size_t>();
        if (config.contains("acfWindowSec")) m_acfWindowSec = std::max(1.0, config["acfWindowSec"].get<double>());
        if (config.contains("historySize")) m_historySize = std::max<size_t>(1, config["historySize"].get<size_t>());
        if (config.contains("octaveCorrection")) m_octaveCorrection = config["octaveCorrection"].get<bool>();
        if (m_cfg.minBPM < 30.f) m_cfg.minBPM = 30.f;
        if (m_cfg.maxBPM > 240.f) m_cfg.maxBPM = 240.f;
        if (m_cfg.minBPM > m_cfg.maxBPM) std::swap(m_cfg.minBPM, m_cfg.maxBPM);
        if (m_cfg.hopSize == 0 || m_cfg.hopSize > m_cfg.frameSize) m_cfg.hopSize = std::max<size_t>(1, m_cfg.frameSize / 4);
        m_history.clear();
        return true;
    }

    void reset() override { m_history.clear(); }

    nlohmann::json process(const core::AudioBuffer& audio, const core::AnalysisContext&) override {
        const float sr = audio.getSampleRate();
        if (audio.getFrameCount() == 0 || audio.getChannelCount() == 0) {
            return makeResultFallback(audio.getDuration());
        }

        std::vector<float> mono = audio.getMono();
        const size_t N = m_cfg.frameSize;
        const size_t H = m_cfg.hopSize;
        if (mono.size() < N || N < 256) return makeResultFallback(audio.getDuration());

        // Window
        const std::vector<float> winF = core::window::hann(N);
        std::vector<double> window(winF.begin(), winF.end());

        const size_t numFrames = 1 + (mono.size() - N) / H;
        if (numFrames < 4) return makeResultFallback(audio.getDuration());

        // FFTW buffers
        double* in = (double*)fftw_malloc(sizeof(double) * N);
        fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
        fftw_plan plan = fftw_plan_dft_r2c_1d(static_cast<int>(N), in, out, FFTW_ESTIMATE);

        std::vector<double> prevLogMag(N / 2 + 1, std::log(1e-10));
        std::vector<double> odf; odf.reserve(numFrames);

        for (size_t f = 0; f < numFrames; ++f) {
            size_t start = f * H;
            for (size_t i = 0; i < N; ++i) in[i] = static_cast<double>(mono[start + i]) * window[i];
            fftw_execute(plan);
            // Build per-bin log magnitude diffs, half-wave rectified
            std::vector<double> diffs; diffs.reserve(N / 2 + 1);
            for (size_t k = 0; k < (N / 2 + 1); ++k) {
                double re = out[k][0];
                double im = out[k][1];
                double mag = std::hypot(re, im);
                double logMag = std::log(mag + 1e-10);
                double d = logMag - prevLogMag[k];
                if (d < 0.0) d = 0.0;
                diffs.push_back(d);
                prevLogMag[k] = logMag;
            }
            // Median aggregation across bins
            if (!diffs.empty()) {
                size_t mid = diffs.size() / 2;
                std::nth_element(diffs.begin(), diffs.begin() + mid, diffs.end());
                odf.push_back(diffs[mid]);
            } else {
                odf.push_back(0.0);
            }
        }

        fftw_destroy_plan(plan);
        fftw_free(in);
        fftw_free(out);

        // ACF via FFT (Wiener-Khinchin) on a recent window
        const double frameRate = sr / static_cast<double>(H);
        size_t winLen = static_cast<size_t>(std::round(m_acfWindowSec * frameRate));
        winLen = std::min(winLen, odf.size());
        if (winLen < 4) return makeResultFallback(audio.getDuration());
        const size_t startIdx = odf.size() - winLen;

        // Prepare real buffer with zero-padding to next power of two * 2
        size_t fftLen = 1;
        while (fftLen < winLen * 2) fftLen <<= 1;
        std::vector<double> x(fftLen, 0.0);
        for (size_t i = 0; i < winLen; ++i) x[i] = odf[startIdx + i];

        // Forward FFT
        fftw_complex* X = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (fftLen));
        fftw_plan fwd = fftw_plan_dft_r2c_1d(static_cast<int>(fftLen), x.data(), X, FFTW_ESTIMATE);
        fftw_execute(fwd);

        // Power spectrum |X|^2 (store back into X as real values, imag=0)
        size_t complexLen = fftLen / 2 + 1;
        for (size_t i = 0; i < complexLen; ++i) {
            double re = X[i][0], im = X[i][1];
            double p = re * re + im * im;
            X[i][0] = p; X[i][1] = 0.0;
        }

        // Inverse FFT to get autocorrelation
        std::vector<double> acfTime(fftLen, 0.0);
        fftw_plan inv = fftw_plan_dft_c2r_1d(static_cast<int>(fftLen), X, acfTime.data(), FFTW_ESTIMATE);
        fftw_execute(inv);
        fftw_destroy_plan(fwd);
        fftw_destroy_plan(inv);
        fftw_free(X);

        // Normalize by fftLen and take first winLen samples
        for (double& v : acfTime) v /= static_cast<double>(fftLen);
        acfTime.resize(winLen);

        // Peak search within BPM-constrained lag range
        int Lmin = static_cast<int>(std::floor(frameRate * 60.0 / m_cfg.maxBPM));
        int Lmax = static_cast<int>(std::ceil (frameRate * 60.0 / m_cfg.minBPM));
        Lmin = std::max(2, Lmin);
        Lmax = std::min<int>(static_cast<int>(winLen - 1), std::max(Lmin + 1, Lmax));
        if (Lmax <= Lmin) return makeResultFallback(audio.getDuration());

        double zeroLag = acfTime[0];
        int bestLag = Lmin; double bestVal = -1e9;
        for (int lag = Lmin; lag <= Lmax; ++lag) {
            double v = acfTime[lag];
            if (v > bestVal) { bestVal = v; bestLag = lag; }
        }
        double confidence = (zeroLag > 1e-9) ? std::clamp(bestVal / zeroLag, 0.0, 1.0) : 0.0;
        if (confidence < 0.1) return makeResultFallback(audio.getDuration());

        double bpm = 60.0 * frameRate / static_cast<double>(bestLag);
        // Optional octave correction
        if (m_octaveCorrection) {
            double bpm2 = bpm * 2.0;
            if (bpm2 <= m_cfg.maxBPM) {
                int lag2 = std::max(Lmin, static_cast<int>(std::round(frameRate * 60.0 / bpm2)));
                if (lag2 >= Lmin && lag2 <= Lmax) {
                    if (acfTime[bestLag] < 1.05 * acfTime[lag2]) { bestLag = lag2; bpm = bpm2; }
                }
            }
        }

        // Stabilize with rolling median
        m_history.push_back(static_cast<float>(bpm));
        if (m_history.size() > m_historySize) m_history.erase(m_history.begin());
        std::vector<float> sorted = m_history;
        std::sort(sorted.begin(), sorted.end());
        double bpmStable = sorted[sorted.size() / 2];

        // Recompute lag and offset using stabilized bpm
        int lagStable = std::max(2, static_cast<int>(std::round(frameRate * 60.0 / bpmStable)));
        double bestScore = -1e9; int bestOffset = 0;
        for (int off = 0; off < lagStable && off < static_cast<int>(odf.size()); ++off) {
            double s = 0.0; for (size_t n = off; n < odf.size(); n += lagStable) s += odf[n];
            if (s > bestScore) { bestScore = s; bestOffset = off; }
        }
        double firstBeatTime = bestOffset / frameRate;
        double beatIntervalSec = 60.0 / bpmStable;

        // Beat grid
        nlohmann::json beatGrid = nlohmann::json::array();
        for (double t = firstBeatTime; t < audio.getDuration(); t += beatIntervalSec) {
            beatGrid.push_back({ {"t", static_cast<float>(t)}, {"strength", 1.0f} });
        }
        nlohmann::json downbeats = nlohmann::json::array();
        for (size_t i = 0; i < beatGrid.size(); i += 4) downbeats.push_back(beatGrid[i]["t"]);

        return makeResult(bpmStable, confidence, static_cast<float>(beatIntervalSec), beatGrid, downbeats);
    }

    bool validateOutput(const nlohmann::json& output) const override {
        return output.contains("bpm") && output.contains("beatGrid") &&
               output["bpm"].is_number() && output["bpm"] >= m_cfg.minBPM && output["bpm"] <= m_cfg.maxBPM;
    }

private:
    BPMConfig m_cfg{};
    double m_acfWindowSec = 8.0; // seconds
    size_t m_historySize = 10;
    bool m_octaveCorrection = true;
    std::vector<float> m_history;

    static nlohmann::json makeResult(double bpm, double conf, float interval,
                                     const nlohmann::json& grid, const nlohmann::json& downbeats) {
        return {
            {"bpm", bpm},
            {"confidence", conf},
            {"beatInterval", interval},
            {"beatGrid", grid},
            {"downbeats", downbeats},
            {"method", "odf-acf-median"}
        };
    }

    nlohmann::json makeResultFallback(double duration) const {
        double bpm = 0.5 * (m_cfg.minBPM + m_cfg.maxBPM);
        double interval = 60.0 / bpm;
        nlohmann::json beatGrid = nlohmann::json::array();
        for (double t = 0.0; t < duration; t += interval) {
            beatGrid.push_back({ {"t", static_cast<float>(t)}, {"strength", 0.0f} });
        }
        nlohmann::json downbeats = nlohmann::json::array();
        for (size_t i = 0; i < beatGrid.size(); i += 4) downbeats.push_back(beatGrid[i]["t"]);
        return makeResult(bpm, 0.0, static_cast<float>(interval), beatGrid, downbeats);
    }
};

std::unique_ptr<core::IAnalysisModule> createRealBPMModule() { return std::make_unique<RealBPMModule>(); }

} // namespace ave::modules
