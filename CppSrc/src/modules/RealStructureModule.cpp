#include "../../include/core/IAnalysisModule.h"
#include "../../include/core/AudioBuffer.h"
#include "../../include/modules/StructureModule.h"
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace ave { namespace modules {

class RealStructureModule : public core::IAnalysisModule {
public:
    std::string getName() const override { return "Structure"; }
    std::string getVersion() const override { return "1.0.0"; }
    bool isRealTime() const override { return false; }

    bool initialize(const nlohmann::json& config) override {
        if (config.contains("segmentMinLength")) m_segmentMinLength = std::max(0.0, config["segmentMinLength"].get<double>());
        if (config.contains("noveltyKernelSize")) m_kernelSize = std::max<int>(4, config["noveltyKernelSize"].get<int>());
        if (config.contains("peakMeanWindow")) m_peakMeanWindow = std::max<int>(4, config["peakMeanWindow"].get<int>());
        if (config.contains("peakThreshold")) m_peakThreshold = config["peakThreshold"].get<double>();
        if (config.contains("debug")) m_debug = config["debug"].get<bool>();
        return true;
    }

    void reset() override {}

    // Depends on spectral features
    std::vector<std::string> getDependencies() const override { return {"Spectral"}; }

    nlohmann::json process(const core::AudioBuffer& audio, const core::AnalysisContext& context) override {
        // Retrieve spectral result
        auto specOpt = context.getModuleResult("Spectral");
        if (!specOpt.has_value()) {
            return makeEmptyResult();
        }
        const nlohmann::json& spec = *specOpt;
        if (!spec.contains("bands")) return makeEmptyResult();
        const auto& bands = spec["bands"];

        // Expected band names
        static const char* NAMES[5] = {"low", "lowMid", "mid", "highMid", "high"};
        for (const char* nm : NAMES) {
            if (!bands.contains(nm)) return makeEmptyResult();
        }

        // Determine number of frames (assume all bands aligned)
        const size_t numFrames = bands["low"].size();
        if (numFrames < static_cast<size_t>(m_kernelSize * 2 + 4)) {
            return makeEmptyResult();
        }

        // Build feature vectors (5D) and L2-normalize per frame
        std::vector<std::vector<double>> features(numFrames, std::vector<double>(5, 0.0));
        for (size_t t = 0; t < numFrames; ++t) {
            double v[5] = {
                bands["low"][t].contains("v") ? bands["low"][t]["v"].get<double>() : 0.0,
                bands["lowMid"][t].contains("v") ? bands["lowMid"][t]["v"].get<double>() : 0.0,
                bands["mid"][t].contains("v") ? bands["mid"][t]["v"].get<double>() : 0.0,
                bands["highMid"][t].contains("v") ? bands["highMid"][t]["v"].get<double>() : 0.0,
                bands["high"][t].contains("v") ? bands["high"][t]["v"].get<double>() : 0.0
            };
            // L2 normalize
            double norm2 = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3] + v[4]*v[4]);
            if (norm2 <= 1e-12) {
                // leave zeros
            } else {
                for (int i = 0; i < 5; ++i) v[i] /= norm2;
            }
            for (int i = 0; i < 5; ++i) features[t][static_cast<size_t>(i)] = v[i];
        }

        // Build SSM (cosine similarity = dot of L2-normalized vectors)
        std::vector<std::vector<double>> ssm(numFrames, std::vector<double>(numFrames, 0.0));
        for (size_t i = 0; i < numFrames; ++i) {
            ssm[i][i] = 1.0;
            for (size_t j = i + 1; j < numFrames; ++j) {
                double dot = 0.0;
                for (int k = 0; k < 5; ++k) dot += features[i][static_cast<size_t>(k)] * features[j][static_cast<size_t>(k)];
                // clamp to [0,1]
                if (dot < 0.0) dot = 0.0; if (dot > 1.0) dot = 1.0;
                ssm[i][j] = dot;
                ssm[j][i] = dot;
            }
        }

        // Novelty curve via checkerboard kernel around diagonal
        const int K = m_kernelSize; // half-size per quadrant
        std::vector<double> novelty(numFrames, 0.0);
        const size_t startT = static_cast<size_t>(K);
        const size_t endT = (numFrames > static_cast<size_t>(K)) ? (numFrames - static_cast<size_t>(K)) : 0;
        const double area = static_cast<double>(K) * static_cast<double>(K);

        for (size_t t = startT; t < endT; ++t) {
            double sumTR = 0.0, sumBL = 0.0, sumTL = 0.0, sumBR = 0.0;
            // top-right: rows [t-K, t-1], cols [t, t+K-1]
            for (int di = -K; di < 0; ++di) {
                size_t i = static_cast<size_t>(static_cast<long long>(t) + di);
                for (int dj = 0; dj < K; ++dj) {
                    size_t j = static_cast<size_t>(static_cast<long long>(t) + dj);
                    sumTR += ssm[i][j];
                }
            }
            // bottom-left: rows [t, t+K-1], cols [t-K, t-1]
            for (int di = 0; di < K; ++di) {
                size_t i = static_cast<size_t>(t + static_cast<size_t>(di));
                for (int dj = -K; dj < 0; ++dj) {
                    size_t j = static_cast<size_t>(static_cast<long long>(t) + dj);
                    sumBL += ssm[i][j];
                }
            }
            // top-left: rows [t-K, t-1], cols [t-K, t-1]
            for (int di = -K; di < 0; ++di) {
                size_t i = static_cast<size_t>(static_cast<long long>(t) + di);
                for (int dj = -K; dj < 0; ++dj) {
                    size_t j = static_cast<size_t>(static_cast<long long>(t) + dj);
                    sumTL += ssm[i][j];
                }
            }
            // bottom-right: rows [t, t+K-1], cols [t, t+K-1]
            for (int di = 0; di < K; ++di) {
                size_t i = static_cast<size_t>(t + static_cast<size_t>(di));
                for (int dj = 0; dj < K; ++dj) {
                    size_t j = static_cast<size_t>(t + static_cast<size_t>(dj));
                    sumBR += ssm[i][j];
                }
            }
            double score = (sumTL + sumBR) - (sumTR + sumBL);
            // Normalize and half-wave rectify to focus on dissimilarity increase at boundaries
            double val = (area > 0.0) ? (score / (2.0 * area)) : score;
            if (val < 0.0) val = 0.0;
            novelty[t] = val;
        }

        // Smooth novelty with small moving average (like Onset)
        std::vector<double> noveltySm(novelty.size(), 0.0);
        int smoothRadius = std::max(4, m_kernelSize / 8);
        for (size_t t = 0; t < novelty.size(); ++t) {
            int a = static_cast<int>(t) - smoothRadius;
            int b = static_cast<int>(t) + smoothRadius;
            a = std::max<int>(0, a);
            b = std::min<int>(static_cast<int>(novelty.size()) - 1, b);
            double sum = 0.0; int cnt = 0;
            for (int i = a; i <= b; ++i) { sum += novelty[static_cast<size_t>(i)]; ++cnt; }
            noveltySm[t] = cnt ? (sum / cnt) : 0.0;
        }

        // Peak picking (reuse logic from RealOnsetModule) with extra global z-score gating
        std::vector<size_t> peakIdx;
        if (noveltySm.size() >= 3) {
            const int W = m_peakMeanWindow;
            // Enforce min distance in frames based on segmentMinLength seconds
            double frameRate = 0.0;
            if (spec.contains("frameRate")) frameRate = spec["frameRate"].get<double>();
            size_t minDistFrames = (frameRate > 0.0) ? static_cast<size_t>(std::floor(m_segmentMinLength * frameRate)) : static_cast<size_t>(W / 2);
            const int minDist = static_cast<int>(std::max<size_t>(static_cast<size_t>(W/2), std::max<size_t>(1, minDistFrames)));
            const int prePost = std::max(1, W / 2);

            // Global stats on novelty for z-score based thresholding
            double gsum = 0.0, gsum2 = 0.0; size_t gcnt = 0;
            for (size_t i = static_cast<size_t>(K); i < noveltySm.size() - static_cast<size_t>(K); ++i) {
                double v = noveltySm[i]; gsum += v; gsum2 += v * v; ++gcnt;
            }
            double gmean = (gcnt ? (gsum / gcnt) : 0.0);
            double gvar = (gcnt ? (gsum2 / gcnt) - gmean * gmean : 0.0);
            if (gvar < 0.0) gvar = 0.0; double gstd = std::sqrt(gvar);

            for (size_t t = 1; t + 1 < noveltySm.size(); ++t) {
                // Skip edge zones to avoid checkerboard kernel edge artifacts
                if (t < static_cast<size_t>(K) || t >= noveltySm.size() - static_cast<size_t>(K)) continue;
                int pmA = static_cast<int>(t) - prePost;
                int pmB = static_cast<int>(t) + prePost;
                pmA = std::max<int>(0, pmA);
                pmB = std::min<int>(static_cast<int>(noveltySm.size()) - 1, pmB);
                bool isMax = true;
                for (int i = pmA; i <= pmB; ++i) {
                    if (noveltySm[static_cast<size_t>(i)] > noveltySm[t]) { isMax = false; break; }
                }
                if (!isMax) continue;
                int a = static_cast<int>(t) - W;
                int b = static_cast<int>(t) - 1;
                a = std::max<int>(0, a);
                b = std::max<int>(a, b);
                double sum = 0.0; int cnt = 0;
                for (int i = a; i <= b; ++i) { sum += noveltySm[static_cast<size_t>(i)]; ++cnt; }
                double lmean = cnt ? (sum / cnt) : 0.0;
                double threshLocal = lmean * (1.0 + m_peakThreshold);
                double threshGlobal = gmean + m_peakThreshold * gstd;
                double thresh = std::max(threshLocal, threshGlobal);
                if (noveltySm[t] >= thresh) {
                    if (!peakIdx.empty()) {
                        if (static_cast<int>(t) - static_cast<int>(peakIdx.back()) < minDist) {
                            if (noveltySm[t] > noveltySm[peakIdx.back()]) {
                                peakIdx.back() = t;
                            }
                            continue;
                        }
                    }
                    peakIdx.push_back(t);
                }
            }
        }

        // Convert peaks to segments: boundaries at peak times; include [0, duration]
        std::vector<double> boundaryTimesSec;
        std::vector<double> boundaryConf;
        const double sr = audio.getSampleRate();
        size_t H = spec.contains("hopSize") ? spec["hopSize"].get<size_t>() : 512;
        for (size_t idx : peakIdx) {
            // compensate smoothing radius to align better with center
            double tSec = static_cast<double>(idx + static_cast<size_t>(smoothRadius)) * static_cast<double>(H) / static_cast<double>(sr);
            boundaryTimesSec.push_back(tSec);
            boundaryConf.push_back(noveltySm[idx]);
        }
        std::sort(boundaryTimesSec.begin(), boundaryTimesSec.end());

        // Build segments array
        nlohmann::json segments = nlohmann::json::array();
        double t0 = 0.0;
        size_t bi = 0;
        for (; bi < boundaryTimesSec.size(); ++bi) {
            double t1 = boundaryTimesSec[bi];
            if (t1 <= t0) continue; // skip degenerate
            double conf = (bi < boundaryConf.size()) ? boundaryConf[bi] : 0.5;
            segments.push_back({{"start", t0}, {"end", t1}, {"label", std::string("segment_") + std::to_string(bi)}, {"confidence", conf}});
            t0 = t1;
        }
        double dur = audio.getDuration();
        if (t0 < dur) {
            double conf = (!boundaryConf.empty()) ? boundaryConf.back() : 0.5;
            segments.push_back({{"start", t0}, {"end", dur}, {"label", std::string("segment_") + std::to_string(bi)}, {"confidence", conf}});
        }

        nlohmann::json result = {
            {"segments", segments},
            {"count", segments.size()}
        };

        if (m_debug) {
            // Provide novelty curve for debugging
            nlohmann::json novArr = nlohmann::json::array();
            for (size_t i = 0; i < noveltySm.size(); ++i) {
                double tSec = static_cast<double>(i) * static_cast<double>(H) / static_cast<double>(sr);
                novArr.push_back({{"t", tSec}, {"v", noveltySm[i]}});
            }
            result["debug"] = nlohmann::json{{"novelty", novArr}};
        }

        return result;
    }

    bool validateOutput(const nlohmann::json& output) const override {
        return output.contains("segments") && output.contains("count");
    }

private:
    double m_segmentMinLength = 8.0; // seconds
    int m_kernelSize = 64;           // frames (half-size per quadrant)
    int m_peakMeanWindow = 32;       // frames
    double m_peakThreshold = 0.1;    // multiplicative increment over mean
    bool m_debug = false;

    static nlohmann::json makeEmptyResult() {
        return nlohmann::json{
            {"segments", nlohmann::json::array()},
            {"count", 0}
        };
    }
};

std::unique_ptr<core::IAnalysisModule> createRealStructureModule() {
    return std::make_unique<RealStructureModule>();
}

} } // namespace ave::modules
