﻿#include "../../include/core/IAnalysisModule.h"
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
        // Optional novelty fusion weights
        if (config.contains("noveltyWeights") && config["noveltyWeights"].is_object()) {
            const auto& w = config["noveltyWeights"];
            if (w.contains("energy"))   m_wE = std::max(0.0, w["energy"].get<double>());
            if (w.contains("mfcc"))     m_wM = std::max(0.0, w["mfcc"].get<double>());
            if (w.contains("chroma"))   m_wC = std::max(0.0, w["chroma"].get<double>());
            if (w.contains("contrast")) m_wX = std::max(0.0, w["contrast"].get<double>());
        } else {
            // Backward-compatible aliases
            if (config.contains("wE")) m_wE = std::max(0.0, config["wE"].get<double>());
            if (config.contains("wM")) m_wM = std::max(0.0, config["wM"].get<double>());
            if (config.contains("wC")) m_wC = std::max(0.0, config["wC"].get<double>());
            if (config.contains("wX")) m_wX = std::max(0.0, config["wX"].get<double>());
        }
        // Sub-segmentation sensitivity controls
        if (config.contains("subPeakThresholdFactor")) {
            m_subPeakThresholdFactor = std::clamp(config["subPeakThresholdFactor"].get<double>(), 0.05, 1.0);
        }
        if (config.contains("subPeakWindowDivisor")) {
            m_subPeakWindowDivisor = std::max(1, config["subPeakWindowDivisor"].get<int>());
        }
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

        // Helper to build cosine SSM for arbitrary feature matrix (frames x dims)
        auto buildCosineSSM = [](const std::vector<std::vector<double>>& F) {
            const size_t T = F.size();
            std::vector<std::vector<double>> S(T, std::vector<double>(T, 0.0));
            for (size_t i = 0; i < T; ++i) {
                S[i][i] = 1.0;
                for (size_t j = i + 1; j < T; ++j) {
                    double dot = 0.0;
                    for (size_t k = 0; k < F[i].size() && k < F[j].size(); ++k) dot += F[i][k] * F[j][k];
                    if (dot < 0.0) dot = 0.0; if (dot > 1.0) dot = 1.0;
                    S[i][j] = dot; S[j][i] = dot;
                }
            }
            return S;
        };
        // Helper to compute novelty from SSM using checkerboard kernel
        auto noveltyFromSSM = [this](const std::vector<std::vector<double>>& S) {
            const int K = m_kernelSize;
            const size_t T = S.size();
            std::vector<double> nov(T, 0.0);
            if (T == 0) return nov;
            const size_t startT = static_cast<size_t>(K);
            const size_t endT = (T > static_cast<size_t>(K)) ? (T - static_cast<size_t>(K)) : 0;
            const double area = static_cast<double>(K) * static_cast<double>(K);
            for (size_t t = startT; t < endT; ++t) {
                double sumTR = 0.0, sumBL = 0.0, sumTL = 0.0, sumBR = 0.0;
                for (int di = -K; di < 0; ++di) {
                    size_t i = static_cast<size_t>(static_cast<long long>(t) + di);
                    for (int dj = 0; dj < K; ++dj) {
                        size_t j = static_cast<size_t>(static_cast<long long>(t) + dj);
                        sumTR += S[i][j];
                    }
                }
                for (int di = 0; di < K; ++di) {
                    size_t i = static_cast<size_t>(t + static_cast<size_t>(di));
                    for (int dj = -K; dj < 0; ++dj) {
                        size_t j = static_cast<size_t>(static_cast<long long>(t) + dj);
                        sumBL += S[i][j];
                    }
                }
                for (int di = -K; di < 0; ++di) {
                    size_t i = static_cast<size_t>(static_cast<long long>(t) + di);
                    for (int dj = -K; dj < 0; ++dj) {
                        size_t j = static_cast<size_t>(static_cast<long long>(t) + dj);
                        sumTL += S[i][j];
                    }
                }
                for (int di = 0; di < K; ++di) {
                    size_t i = static_cast<size_t>(t + static_cast<size_t>(di));
                    for (int dj = 0; dj < K; ++dj) {
                        size_t j = static_cast<size_t>(t + static_cast<size_t>(dj));
                        sumBR += S[i][j];
                    }
                }
                double score = (sumTL + sumBR) - (sumTR + sumBL);
                double val = (area > 0.0) ? (score / (2.0 * area)) : score;
                if (val < 0.0) val = 0.0;
                nov[t] = val;
            }
            return nov;
        };

        // Energy SSM and novelty
        std::vector<std::vector<double>> ssmE = buildCosineSSM(features);
        std::vector<double> noveltyE = noveltyFromSSM(ssmE);

        // MFCC features (optional)
        std::vector<double> noveltyMAPPED_MFCC(numFrames, 0.0);
        if (spec.contains("mfcc") && spec["mfcc"].is_array() && !spec["mfcc"].empty()) {
            size_t mFrames = spec["mfcc"].size();
            std::vector<std::vector<double>> mfccFeat(mFrames);
            for (size_t i = 0; i < mFrames; ++i) {
                const auto& fr = spec["mfcc"][i];
                if (fr.contains("v") && fr["v"].is_array()) {
                    const auto& arr = fr["v"];
                    size_t C = arr.size();
                    mfccFeat[i].assign(C, 0.0);
                    double norm2 = 0.0;
                    for (size_t c = 0; c < C; ++c) { double vv = arr[c].get<double>(); mfccFeat[i][c] = vv; norm2 += vv * vv; }
                    if (norm2 > 1e-12) { double inv = 1.0 / std::sqrt(norm2); for (double& v : mfccFeat[i]) v *= inv; }
                } else {
                    mfccFeat[i] = std::vector<double>(13, 0.0);
                }
            }
            if (mFrames >= static_cast<size_t>(m_kernelSize * 2 + 4)) {
                auto ssmM = buildCosineSSM(mfccFeat);
                auto novM = noveltyFromSSM(ssmM);
                // Map MFCC novelty to spectral timeline by nearest index
                for (size_t t = 0; t < numFrames; ++t) {
                    size_t idx = static_cast<size_t>(std::llround((static_cast<double>(t) * static_cast<double>(mFrames)) / std::max<size_t>(1, numFrames)));
                    if (idx >= novM.size()) idx = novM.size() - 1;
                    noveltyMAPPED_MFCC[t] = novM[idx];
                }
            }
        }

        // Chroma features (optional from Tonality)
        std::vector<double> noveltyMAPPED_CHROMA(numFrames, 0.0);
        if (auto tonOpt = context.getModuleResult("Tonality"); tonOpt && (*tonOpt).contains("chromaSequence") && (*tonOpt)["chromaSequence"].is_array()) {
            const auto& chromaSeq = (*tonOpt)["chromaSequence"];
            size_t cFrames = chromaSeq.size();
            if (cFrames >= static_cast<size_t>(m_kernelSize * 2 + 4)) {
                std::vector<std::vector<double>> chromaFeat(cFrames, std::vector<double>(12, 0.0));
                for (size_t i = 0; i < cFrames; ++i) {
                    const auto& fr = chromaSeq[i];
                    if (fr.contains("v") && fr["v"].is_array()) {
                        double norm2 = 0.0;
                        for (size_t k = 0; k < 12 && k < fr["v"].size(); ++k) { double vv = fr["v"][k].get<double>(); chromaFeat[i][k] = vv; norm2 += vv * vv; }
                        if (norm2 > 1e-12) { double inv = 1.0 / std::sqrt(norm2); for (double& v : chromaFeat[i]) v *= inv; }
                    }
                }
                auto ssmC = buildCosineSSM(chromaFeat);
                auto novC = noveltyFromSSM(ssmC);
                for (size_t t = 0; t < numFrames; ++t) {
                    size_t idx = static_cast<size_t>(std::llround((static_cast<double>(t) * static_cast<double>(cFrames)) / std::max<size_t>(1, numFrames)));
                    if (idx >= novC.size()) idx = novC.size() - 1;
                    noveltyMAPPED_CHROMA[t] = novC[idx];
                }
            }
        }

        // Spectral Contrast features (optional)
        std::vector<double> noveltyMAPPED_CONTRAST(numFrames, 0.0);
        if (spec.contains("spectralContrast") && spec["spectralContrast"].is_array() && !spec["spectralContrast"].empty()) {
            size_t cFrames = spec["spectralContrast"].size();
            std::vector<std::vector<double>> contrastFeat(cFrames);
            for (size_t i = 0; i < cFrames; ++i) {
                const auto& fr = spec["spectralContrast"][i];
                if (fr.contains("v") && fr["v"].is_array()) {
                    size_t D = fr["v"].size();
                    contrastFeat[i].assign(D, 0.0);
                    double norm2 = 0.0;
                    for (size_t d = 0; d < D; ++d) { double vv = fr["v"][d].get<double>(); contrastFeat[i][d] = vv; norm2 += vv * vv; }
                    if (norm2 > 1e-12) { double inv = 1.0 / std::sqrt(norm2); for (double& v : contrastFeat[i]) v *= inv; }
                } else {
                    contrastFeat[i] = std::vector<double>(6, 0.0);
                }
            }
            if (cFrames >= static_cast<size_t>(m_kernelSize * 2 + 4)) {
                auto ssmX = buildCosineSSM(contrastFeat);
                auto novX = noveltyFromSSM(ssmX);
                // Map contrast novelty to spectral timeline by nearest index
                for (size_t t = 0; t < numFrames; ++t) {
                    size_t idx = static_cast<size_t>(std::llround((static_cast<double>(t) * static_cast<double>(cFrames)) / std::max<size_t>(1, numFrames)));
                    if (idx >= novX.size()) idx = novX.size() - 1;
                    noveltyMAPPED_CONTRAST[t] = novX[idx];
                }
            }
        }

        // Normalize each novelty to [0,1]
        auto normalize = [](std::vector<double>& v) {
            double mx = 0.0; for (double x : v) if (x > mx) mx = x; if (mx > 0.0) for (double& x : v) x /= mx;
        };
        normalize(noveltyE);
        normalize(noveltyMAPPED_MFCC);
        normalize(noveltyMAPPED_CHROMA);
        normalize(noveltyMAPPED_CONTRAST);

        // Fuse novelty curves with weights (E=0.2, MFCC=0.3, Chroma=0.3, Contrast=0.2)
        std::vector<double> novelty(numFrames, 0.0);
        double sumW = 0.0;
        // determine availability
        bool hasE = true;
        bool hasM = false; for (double v : noveltyMAPPED_MFCC) { if (v > 0.0) { hasM = true; break; } }
        bool hasC = false; for (double v : noveltyMAPPED_CHROMA) { if (v > 0.0) { hasC = true; break; } }
        bool hasX = false; for (double v : noveltyMAPPED_CONTRAST) { if (v > 0.0) { hasX = true; break; } }
        if (hasE) sumW += m_wE; if (hasM) sumW += m_wM; if (hasC) sumW += m_wC; if (hasX) sumW += m_wX; if (sumW <= 0.0) sumW = 1.0;
        for (size_t t = 0; t < numFrames; ++t) {
            double s = 0.0;
            if (hasE) s += m_wE * noveltyE[t];
            if (hasM) s += m_wM * noveltyMAPPED_MFCC[t];
            if (hasC) s += m_wC * noveltyMAPPED_CHROMA[t];
            if (hasX) s += m_wX * noveltyMAPPED_CONTRAST[t];
            novelty[t] = s / sumW;
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
            const int K = m_kernelSize;
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

        // Optional beat snapping: snap boundaries to nearest downbeat from BPM module if available
        if (auto bpmOpt = context.getModuleResult("BPM"); bpmOpt && (*bpmOpt).contains("downbeats") && (*bpmOpt)["downbeats"].is_array()) {
            const auto& downbeats = (*bpmOpt)["downbeats"];
            if (!downbeats.empty()) {
                for (double& bt : boundaryTimesSec) {
                    // Find nearest downbeat
                    double bestT = bt;
                    double bestD = 1e12;
                    for (const auto& db : downbeats) {
                        double dbt = db.get<double>();
                        double d = std::abs(dbt - bt);
                        if (d < bestD) { bestD = d; bestT = dbt; }
                    }
                    bt = bestT;
                }
                // Re-sort and deduplicate very close boundaries after snapping
                std::sort(boundaryTimesSec.begin(), boundaryTimesSec.end());
                std::vector<double> snapped;
                const double minGap = 0.1; // 100 ms minimum gap between boundaries
                for (double t : boundaryTimesSec) {
                    if (snapped.empty() || std::abs(t - snapped.back()) >= minGap) {
                        snapped.push_back(t);
                    }
                }
                boundaryTimesSec.swap(snapped);
            }
        }

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

        // Phase 2.2: Hierarchical analysis using global novelty (refactored)
        performHierarchicalAnalysis(segments, noveltySm, spec, context, audio);

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
    // Novelty fusion weights (configurable)
    double m_wE = 0.2; // Energy
    double m_wM = 0.3; // MFCC
    double m_wC = 0.3; // Chroma
    double m_wX = 0.2; // Spectral Contrast
    // Sub-segmentation sensitivity controls
    double m_subPeakThresholdFactor = 0.5; // thrSub = m_peakThreshold * factor
    int m_subPeakWindowDivisor = 2;        // Wsub = m_peakMeanWindow / divisor

    // Phase 2.2: Multi-Scale Peak Picking on global novelty curve
    void performHierarchicalAnalysis(
        nlohmann::json& segments,
        const std::vector<double>& noveltySm,
        const nlohmann::json& spec,
        const core::AnalysisContext& context,
        const core::AudioBuffer& audio
    ) const {
        if (!spec.contains("frameRate")) {
            for (auto& seg : segments) seg["sub_segments"] = nlohmann::json::array();
            return;
        }
        double frameRate = spec["frameRate"].get<double>();
        size_t Hspec = spec.contains("hopSize") ? spec["hopSize"].get<size_t>() : 512;
        double srLoc = audio.getSampleRate();

        const double primaryThreshold = m_peakThreshold;
        const int primaryWindow = m_peakMeanWindow;

        const double thrSub = std::max(0.01, primaryThreshold * m_subPeakThresholdFactor);
        const int Wsub = std::max(4, primaryWindow / std::max(1, m_subPeakWindowDivisor));

        const int smoothRadius = std::max(4, m_kernelSize / 8);

        std::vector<double> downbeatsVec;
        if (auto bpmOpt = context.getModuleResult("BPM"); bpmOpt && (*bpmOpt).contains("downbeats") && (*bpmOpt)["downbeats"].is_array()) {
            for (const auto& db : (*bpmOpt)["downbeats"]) downbeatsVec.push_back(db.get<double>());
        }
        auto snapTimes = [&downbeatsVec](std::vector<double>& times){
            if (downbeatsVec.empty()) return;
            for (double& bt : times) {
                double bestT = bt; double bestD = 1e12;
                for (double dbt : downbeatsVec) { double d = std::abs(dbt - bt); if (d < bestD) { bestD = d; bestT = dbt; } }
                bt = bestT;
            }
            std::sort(times.begin(), times.end());
            std::vector<double> tmp;
            const double minGap=0.1;
            for (double t : times){ if (tmp.empty() || std::abs(t - tmp.back()) >= minGap) tmp.push_back(t);}
            times.swap(tmp);
        };

        auto findPeaksOnCurve = [&](const std::vector<double>& cur) {
            std::vector<size_t> peaks;
            if (cur.size() < 3) return peaks;
            double gsum = 0.0, gsum2 = 0.0; size_t gcnt = 0;
            for (double v : cur) { gsum += v; gsum2 += v*v; ++gcnt; }
            double gmean = (gcnt ? (gsum / gcnt) : 0.0);
            double gvar = (gcnt ? (gsum2 / gcnt) - gmean * gmean : 0.0); if (gvar < 0.0) gvar = 0.0; double gstd = std::sqrt(gvar);
            size_t minDistFrames = (frameRate > 0.0) ? static_cast<size_t>(std::floor(m_segmentMinLength * frameRate)) : static_cast<size_t>(Wsub / 2);
            const int minDist = static_cast<int>(std::max<size_t>(static_cast<size_t>(Wsub/2), std::max<size_t>(1, minDistFrames)));
            const int prePostSub = std::max(1, Wsub / 2);
            for (size_t t = 1; t + 1 < cur.size(); ++t) {
                int pmA = static_cast<int>(t) - prePostSub;
                int pmB = static_cast<int>(t) + prePostSub;
                pmA = std::max<int>(0, pmA);
                pmB = std::min<int>(static_cast<int>(cur.size()) - 1, pmB);
                bool isMax = true;
                for (int i = pmA; i <= pmB; ++i) { if (cur[static_cast<size_t>(i)] > cur[t]) { isMax = false; break; } }
                if (!isMax) continue;
                int a = static_cast<int>(t) - Wsub; int b = static_cast<int>(t) - 1;
                a = std::max<int>(0, a); b = std::max<int>(a, b);
                double sum = 0.0; int cnt = 0; for (int i = a; i <= b; ++i) { sum += cur[static_cast<size_t>(i)]; ++cnt; }
                double lmean = cnt ? (sum / cnt) : 0.0;
                double threshLocal = lmean * (1.0 + thrSub);
                double threshGlobal = gmean + thrSub * gstd;
                double thresh = std::max(threshLocal, threshGlobal);
                if (cur[t] >= thresh) {
                    if (!peaks.empty()) {
                        if (static_cast<int>(t) - static_cast<int>(peaks.back()) < minDist) {
                            if (cur[t] > cur[peaks.back()]) { peaks.back() = t; }
                            continue;
                        }
                    }
                    peaks.push_back(t);
                }
            }
            return peaks;
        };

        for (auto& seg : segments) {
            double s = seg.value("start", 0.0);
            double e = seg.value("end", s);
            size_t i0 = static_cast<size_t>(std::floor(s * frameRate));
            size_t i1 = static_cast<size_t>(std::ceil(e * frameRate));
            if (i1 <= i0 + 2) { seg["sub_segments"] = nlohmann::json::array(); continue; }
            i0 = std::min(i0, noveltySm.size());
            i1 = std::min(i1, noveltySm.size());
            if (i1 <= i0 + 2) { seg["sub_segments"] = nlohmann::json::array(); continue; }
            std::vector<double> subNoveltyCurve(noveltySm.begin() + static_cast<std::ptrdiff_t>(i0), noveltySm.begin() + static_cast<std::ptrdiff_t>(i1));
            std::vector<size_t> subPeaks = findPeaksOnCurve(subNoveltyCurve);
            std::vector<double> localTimes; localTimes.reserve(subPeaks.size());
            for (size_t idx : subPeaks) {
                double tSec = static_cast<double>(i0 + idx + static_cast<size_t>(smoothRadius)) * static_cast<double>(Hspec) / static_cast<double>(srLoc);
                if (tSec > s && tSec < e) localTimes.push_back(tSec);
            }
            std::sort(localTimes.begin(), localTimes.end());
            snapTimes(localTimes);
            nlohmann::json subs = nlohmann::json::array();
            double tStart = s; size_t phraseCount = 0;
            for (double tEnd : localTimes) {
                if (tEnd <= tStart) continue;
                size_t idx = static_cast<size_t>(std::min<double>(std::max(0.0, (tEnd - s) * frameRate), static_cast<double>(subNoveltyCurve.size() - 1)));
                double conf = subNoveltyCurve[idx];
                subs.push_back({{"start", tStart}, {"end", tEnd}, {"label", std::string("phrase_") + std::to_string(++phraseCount)}, {"confidence", conf}});
                tStart = tEnd;
            }
            if (tStart < e) {
                double conf = (subNoveltyCurve.empty() ? 0.5 : subNoveltyCurve.back());
                subs.push_back({{"start", tStart}, {"end", e}, {"label", std::string("phrase_") + std::to_string(++phraseCount)}, {"confidence", conf}});
            }
            seg["sub_segments"] = subs;
        }
    }

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
