#include "../../include/core/IAnalysisModule.h"
#include "../../include/core/AudioBuffer.h"
#include "../../include/modules/OnsetModule.h"
#include <nlohmann/json.hpp>
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

    std::vector<std::string> getDependencies() const override { return {"BPM"}; }

    nlohmann::json process(const core::AudioBuffer& audio, const core::AnalysisContext& context) override {
        // 1. Retrieve ODF from BPM module result
        nlohmann::json odfJson = nlohmann::json::array();
        auto bpmResultOpt = context.getModuleResult("BPM");
        if (bpmResultOpt && (*bpmResultOpt).contains("internal") && (*bpmResultOpt)["internal"].contains("odf") && (*bpmResultOpt)["internal"]["odf"].is_array() && !(*bpmResultOpt)["internal"]["odf"].empty()) {
            odfJson = (*bpmResultOpt)["internal"]["odf"];
        } else {
            // Compatibility fallback: compute a simple energy-based ODF directly
            // using Onset module settings (time-domain, no FFT)
            const size_t N = m_fftSize;
            const size_t H = m_hopSize == 0 ? std::max<size_t>(1, N / 4) : m_hopSize;
            const double sr = context.sampleRate;
            std::vector<float> mono = audio.getMono();
            if (!mono.empty()) {
                // Hann window for energy computation
                std::vector<float> winF = core::window::hann(N);
                std::vector<double> window(winF.begin(), winF.end());
                const size_t numFrames = (mono.size() + H - 1) / H;
                std::vector<double> energy(numFrames, 0.0);
                for (size_t f = 0; f < numFrames; ++f) {
                    const size_t start = f * H;
                    double e = 0.0;
                    for (size_t i = 0; i < N; ++i) {
                        size_t idx = start + i;
                        double s = (idx < mono.size()) ? static_cast<double>(mono[idx]) : 0.0;
                        e += (s * s) * window[i];
                    }
                    energy[f] = e;
                }
                // Half-wave rectified energy difference as ODF
                std::vector<double> odfVals; odfVals.resize(energy.size(), 0.0);
                for (size_t t = 1; t < energy.size(); ++t) {
                    double d = energy[t] - energy[t - 1];
                    if (d > 0.0) odfVals[t] = d; else odfVals[t] = 0.0;
                }
                // Build JSON [{t,v}]
                odfJson = nlohmann::json::array();
                for (size_t i = 0; i < odfVals.size(); ++i) {
                    double tSec = static_cast<double>(i) * (static_cast<double>(H) / sr);
                    odfJson.push_back({{"t", tSec}, {"v", odfVals[i]}});
                }
            }
        }

        // 2. Convert JSON to raw ODF values
        std::vector<double> odfValues;
        odfValues.reserve(odfJson.size());
        for (const auto& frame : odfJson) {
            if (frame.contains("v")) odfValues.push_back(frame["v"].get<double>());
            else odfValues.push_back(0.0);
        }

        // Optional smoothing to reduce ripples
        std::vector<double> odfSm(odfValues.size(), 0.0);
        int smoothRadius = 4; // 9-point moving average
        if (!odfValues.empty()) {
            for (size_t t = 0; t < odfValues.size(); ++t) {
                int a = static_cast<int>(t) - smoothRadius;
                int b = static_cast<int>(t) + smoothRadius;
                a = std::max<int>(0, a);
                b = std::min<int>(static_cast<int>(odfValues.size()) - 1, b);
                double sum = 0.0; int cnt = 0;
                for (int i = a; i <= b; ++i) { sum += odfValues[static_cast<size_t>(i)]; ++cnt; }
                odfSm[t] = cnt ? (sum / cnt) : 0.0;
            }
        }

        // 3. Peak picking on smoothed ODF
        std::vector<size_t> peakIdx;
        if (odfSm.size() >= 3) {
            const int W = m_peakMeanWindow;
            const int minDist = std::max(1, W / 2);
            const int prePost = std::max(1, W / 2);
            for (size_t t = 1; t + 1 < odfSm.size(); ++t) {
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
                int b = static_cast<int>(t) - 1;
                a = std::max<int>(0, a);
                b = std::max<int>(a, b);
                double sum = 0.0; int cnt = 0;
                for (int i = a; i <= b; ++i) { sum += odfSm[static_cast<size_t>(i)]; ++cnt; }
                double mean = cnt ? (sum / cnt) : 0.0;
                if (mean < 1e-6) continue;
                double mult = (1.0 + (m_sensitivity * m_peakThreshold));
                double thresh = mean * mult;
                if (odfSm[t] >= thresh) {
                    if (!peakIdx.empty()) {
                        if (static_cast<int>(t) - static_cast<int>(peakIdx.back()) < minDist) {
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

        // 4. Build output using timestamps from ODF JSON
        nlohmann::json onsets = nlohmann::json::array();
        double framePeriod = 0.0;
        if (odfJson.size() >= 2 && odfJson[0].contains("t") && odfJson[1].contains("t")) {
            framePeriod = odfJson[1]["t"].get<double>() - odfJson[0]["t"].get<double>();
            if (framePeriod < 0.0) framePeriod = 0.0;
        }
        const double timeComp = static_cast<double>(smoothRadius) * framePeriod;
        for (size_t idx : peakIdx) {
            double tSec = odfJson[idx].contains("t") ? odfJson[idx]["t"].get<double>() : static_cast<double>(idx) * framePeriod;
            tSec += timeComp;
            onsets.push_back({ {"t", tSec}, {"strength", odfSm[idx]} });
        }

        return {
            {"onsets", onsets},
            {"count", onsets.size()},
            {"sensitivity", m_sensitivity}
        };
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