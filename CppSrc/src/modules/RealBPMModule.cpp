#include "../../include/modules/BPMModule.h"
#include "../../include/core/IAnalysisModule.h"
#include "../../include/core/AudioBuffer.h"
#include <nlohmann/json.hpp>
#include <fftw3.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <complex>
#include <string>
#include <cctype>
// QM-DSP (Queen Mary) headers for Mixxx-identical engine
#include <dsp/onsets/DetectionFunction.h>
#include <dsp/tempotracking/TempoTrackV2.h>
#include <maths/MathUtilities.h>

namespace ave::modules {

class RealBPMModule : public core::IAnalysisModule {
public:
    std::string getName() const override { return "BPM"; }
    std::string getVersion() const override { return "2.0.0-beattracking"; }

    bool initialize(const nlohmann::json& config) override {
        if (config.contains("minBPM")) m_cfg.minBPM = config["minBPM"].get<float>();
        if (config.contains("maxBPM")) m_cfg.maxBPM = config["maxBPM"].get<float>();
        if (config.contains("frameSize")) m_cfg.frameSize = config["frameSize"].get<size_t>();
        if (config.contains("hopSize")) m_cfg.hopSize = config["hopSize"].get<size_t>();
        if (config.contains("acfWindowSec")) m_acfWindowSec = std::max(1.0, config["acfWindowSec"].get<double>());
        if (config.contains("historySize")) m_historySize = std::max<size_t>(1, config["historySize"].get<size_t>());
        if (config.contains("octaveCorrection")) m_octaveCorrection = config["octaveCorrection"].get<bool>();
        if (config.contains("qmLike")) m_qmLike = config["qmLike"].get<bool>();
        if (config.contains("fixedTempo")) m_fixedTempo = config["fixedTempo"].get<bool>();
        if (config.contains("fastAnalysisSeconds")) {
            double fas = config["fastAnalysisSeconds"].get<double>();
            if (fas > 0.0) {
                // Clamp to [10,45] seconds when enabled
                m_fastAnalysisSec = std::max(10.0, std::min(45.0, fas));
            } else {
                m_fastAnalysisSec = 0.0;
            }
        }
        // R4: Hybrid tempogram configuration
        if (config.contains("hybridTempogram")) m_hybridTempogram = config["hybridTempogram"].get<bool>();
        if (config.contains("combLambda")) {
            double lam = config["combLambda"].get<double>();
            if (lam < 0.0) lam = 0.0; if (lam > 1.0) lam = 1.0;
            m_combLambda = lam;
        }
        if (config.contains("combHarmonics")) {
            int Hc = config["combHarmonics"].get<int>();
            if (Hc < 2) Hc = 2; if (Hc > 8) Hc = 8;
            m_combHarmonics = Hc;
        }
        // Engine selection: allow exact QM-DSP path
        if (config.contains("engine")) {
            std::string eng = config["engine"].get<std::string>();
            std::string elow = eng;
            std::transform(elow.begin(), elow.end(), elow.begin(), [](unsigned char c){return (char)std::tolower(c);} );
            if (elow == "qm" || elow == "qm-dsp" || elow == "queen-mary") {
                m_useQMDSP = true;
            } else if (elow == "native") {
                m_useQMDSP = false;
            }
        } else if (config.contains("useQMDSP")) {
            m_useQMDSP = config["useQMDSP"].get<bool>();
        }
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
        m_octaveSwitchedLast = false; // reset health flag per track
        if (audio.getFrameCount() == 0 || audio.getChannelCount() == 0) {
            return makeResultFallback(audio.getDuration());
        }

        std::vector<float> mono = audio.getMono();

        // If exact QM-DSP engine is requested, use it and bypass native pipeline
        if (m_useQMDSP) {
            std::vector<double> beatTimes = processWithQMDSP(mono, sr, audio.getDuration());
            if (beatTimes.empty()) return makeResultFallback(audio.getDuration());
            nlohmann::json bpmResult = generateBeatTrackingResult(beatTimes, audio.getDuration());
            // Attach ODF from QM-DSP path if available
            nlohmann::json odfJson = nlohmann::json::array();
            if (!m_lastODF.empty() && m_lastODFFrameRate > 0.0) {
                for (size_t i = 0; i < m_lastODF.size(); ++i) {
                    double tSec = static_cast<double>(i) / m_lastODFFrameRate;
                    odfJson.push_back({{"t", tSec}, {"v", m_lastODF[i]}});
                }
            }
            bpmResult["internal"] = {{"odf", odfJson}};
            return bpmResult;
        }

        size_t N = m_cfg.frameSize;
        size_t H = m_cfg.hopSize;
        if (m_qmLike) {
            size_t hopQM = static_cast<size_t>(std::llround(sr * 0.01161));
            if (hopQM < 1) hopQM = 1;
            auto nextPow2 = [](size_t x){ size_t p = 1; while (p < x) p <<= 1; return p; };
            size_t frameQM = nextPow2(static_cast<size_t>(std::ceil(sr / 50.0)));
            if (frameQM < 256) frameQM = 256;
            if (hopQM > frameQM) hopQM = std::max<size_t>(1, frameQM / 4);
            N = frameQM;
            H = hopQM;
        }
        // Fast analysis: optionally limit analysis to the first m_fastAnalysisSec seconds
        if (m_fastAnalysisSec > 0.0) {
            size_t maxSamples = static_cast<size_t>(std::llround(std::min<double>(mono.size(), m_fastAnalysisSec * sr)));
            if (maxSamples >= N) {
                mono.resize(maxSamples);
            }
        }
        if (mono.size() < N || N < 256) return makeResultFallback(audio.getDuration());

        const double frameRate = sr / static_cast<double>(H);

        // NEW BEAT TRACKING PIPELINE

        // STEP 1: Advanced ODF Extraction (Complex Spectral Difference)
        std::vector<double> odf = extractComplexSpectralDifferenceODF(mono, N, H, sr);
        if (odf.size() < 10) return makeResultFallback(audio.getDuration());

        // STEP 2: Peak Detection to find beat candidates
        std::vector<BeatCandidate> beatCandidates = detectBeatCandidates(odf, frameRate);
        if (beatCandidates.empty()) return makeResultFallback(audio.getDuration());

        // STEP 3: Tempogram Analysis
        std::vector<std::vector<double>> tempogram = computeTempogram(odf, frameRate);

        // STEP 4: Dynamic Programming (Viterbi-style) Beat Tracking
        std::vector<double> beatTimes = trackBeatsWithDynamicProgramming(beatCandidates, tempogram, frameRate, audio.getDuration());

        if (beatTimes.empty()) return makeResultFallback(audio.getDuration());

        // STEP 5: Convert beat times to BPM and generate output
        // Expose ODF for downstream modules
        m_lastODF = odf;
        m_lastODFFrameRate = frameRate;
        nlohmann::json bpmResult = generateBeatTrackingResult(beatTimes, audio.getDuration());
        nlohmann::json odfJson = nlohmann::json::array();
        for (size_t i = 0; i < odf.size(); ++i) {
            double tSec = static_cast<double>(i) / frameRate;
            odfJson.push_back({{"t", tSec}, {"v", odf[i]}});
        }
        bpmResult["internal"] = {{"odf", odfJson}};
        return bpmResult;
    }

    bool validateOutput(const nlohmann::json& output) const override {
        return output.contains("bpm") && output.contains("beatGrid") &&
               output["bpm"].is_number() && output["bpm"] >= m_cfg.minBPM && output["bpm"] <= m_cfg.maxBPM;
    }

private:
    // Beat candidate structure for dynamic programming
    struct BeatCandidate {
        double time;        // Time in seconds
        double strength;    // ODF peak strength
        size_t frameIndex;  // Frame index in ODF

        BeatCandidate(double t, double s, size_t idx) : time(t), strength(s), frameIndex(idx) {}
    };

    BPMConfig m_cfg{};
    double m_acfWindowSec = 8.0; // seconds (kept for compatibility)
    size_t m_historySize = 10;
    bool m_octaveCorrection = true;
    bool m_qmLike = false;
    bool m_fixedTempo = false;
    // Exact Mixxx/QM-DSP engine toggle
    bool m_useQMDSP = false;
    double m_fastAnalysisSec = 0.0;
    // R4: Hybrid tempogram (ACF + comb-like evidence)
    bool m_hybridTempogram = false;
    double m_combLambda = 0.3; // blend weight for COMB in [0,1]
    int m_combHarmonics = 4;   // number of harmonics for comb evidence
    // C2: Health flags support
    bool m_octaveSwitchedLast = false; // set true if octave correction selected an alternate grid
    std::vector<float> m_history;

    // Exposed ODF for downstream modules (e.g., Onset)
    std::vector<double> m_lastODF;
    double m_lastODFFrameRate = 0.0; // frames per second for m_lastODF

    // New beat tracking methods
    std::vector<double> extractComplexSpectralDifferenceODF(const std::vector<float>& mono, size_t N, size_t H, float sr);
    std::vector<BeatCandidate> detectBeatCandidates(const std::vector<double>& odf, double frameRate);
    std::vector<std::vector<double>> computeTempogram(const std::vector<double>& odf, double frameRate);
    std::vector<double> trackBeatsWithDynamicProgramming(const std::vector<BeatCandidate>& candidates,
                                                       const std::vector<std::vector<double>>& tempogram,
                                                       double frameRate, double duration);
    nlohmann::json generateBeatTrackingResult(const std::vector<double>& beatTimes, double duration);

    // QM-DSP direct path (Mixxx-identical engine)
    std::vector<double> processWithQMDSP(const std::vector<float>& mono, float sr, double duration);

    // Helper methods for dynamic programming
    double computeTransitionCost(const BeatCandidate& prev, const BeatCandidate& current,
                               const std::vector<std::vector<double>>& tempogram, double frameRate);
    void fillMissingBeats(std::vector<double>& beatTimes, double duration);
    std::vector<double> postProcessOctaveCorrection(const std::vector<double>& beatTimes,
                                                    double duration,
                                                    const std::vector<std::vector<double>>& tempogram);

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

// Implementation of complex spectral difference ODF (Davies & Plumbley approach)
std::vector<double> RealBPMModule::extractComplexSpectralDifferenceODF(const std::vector<float>& mono, size_t N, size_t H, float sr) {
    const std::vector<float> winF = core::window::hann(N);
    std::vector<double> window(winF.begin(), winF.end());

    const size_t numFrames = 1 + (mono.size() - N) / H;
    std::vector<double> odf; odf.reserve(numFrames);

    // FFTW buffers - Fixed C-style cast to static_cast
    double* in = static_cast<double*>(fftw_malloc(sizeof(double) * N));
    fftw_complex* out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1)));
    fftw_plan plan = fftw_plan_dft_r2c_1d(static_cast<int>(N), in, out, FFTW_ESTIMATE);

    // Store previous frame's complex spectrum for phase difference
    std::vector<std::complex<double>> prevSpectrum(N / 2 + 1, std::complex<double>(1e-10, 0.0));

    for (size_t f = 0; f < numFrames; ++f) {
        size_t start = f * H;

        // Apply window
        for (size_t i = 0; i < N; ++i) {
            in[i] = static_cast<double>(mono[start + i]) * window[i];
        }

        fftw_execute(plan);

        // Compute complex spectral difference
        double complexDiff = 0.0;
        for (size_t k = 0; k < (N / 2 + 1); ++k) {
            std::complex<double> current(out[k][0], out[k][1]);
            std::complex<double> diff = current - prevSpectrum[k];

            // Complex spectral difference: magnitude of the difference
            double diffMag = std::abs(diff);

            // Half-wave rectification: only positive changes contribute
            if (std::abs(current) > std::abs(prevSpectrum[k])) {
                complexDiff += diffMag;
            }

            prevSpectrum[k] = current;
        }

        // Normalize by number of bins
        odf.push_back(complexDiff / (N / 2 + 1));
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    // Apply smoothing to reduce noise
    std::vector<double> smoothedODF = odf;
    for (size_t i = 1; i < smoothedODF.size() - 1; ++i) {
        smoothedODF[i] = 0.25 * odf[i-1] + 0.5 * odf[i] + 0.25 * odf[i+1];
    }

    return smoothedODF;
}

// Peak detection for beat candidates
std::vector<RealBPMModule::BeatCandidate> RealBPMModule::detectBeatCandidates(const std::vector<double>& odf, double frameRate) {
    std::vector<BeatCandidate> candidates;

    if (odf.size() < 5) return candidates;

    // Adaptive threshold based on local statistics
    const size_t windowSize = static_cast<size_t>(frameRate * 0.5); // ~500ms window
    std::vector<double> localMean(odf.size());
    std::vector<double> localStd(odf.size());

    // Compute local statistics
    for (size_t i = 0; i < odf.size(); ++i) {
        size_t start = (i >= windowSize/2) ? i - windowSize/2 : 0;
        size_t end = std::min(i + windowSize/2 + 1, odf.size());

        double sum = 0.0, sumSq = 0.0;
        for (size_t j = start; j < end; ++j) {
            sum += odf[j];
            sumSq += odf[j] * odf[j];
        }
        size_t count = end - start;
        localMean[i] = sum / count;
        localStd[i] = std::sqrt(std::max(0.0, sumSq/count - localMean[i]*localMean[i]));
    }

    // Non-maximum suppression within a minimum interval window
    const double minPeakInterval = 60.0 / 300.0; // Minimum 300 BPM (maximum tempo)
    const size_t minPeakFrames = static_cast<size_t>(minPeakInterval * frameRate);

    for (size_t i = 2; i + 2 < odf.size(); ++i) {
        // Check for strict local maximum over a 5-sample neighborhood
        if (odf[i] > odf[i-1] && odf[i] > odf[i+1] && odf[i] > odf[i-2] && odf[i] > odf[i+2]) {
            // Adaptive threshold: mean + 1.0 * std (more sensitive than previous 1.5)
            double threshold = localMean[i] + 1.0 * localStd[i];
            threshold = std::max(threshold, 0.01);

            if (odf[i] > threshold) {
                bool suppressed = false;
                // If a previous candidate is too close in time, keep the stronger one
                if (!candidates.empty()) {
                    size_t lastIdx = candidates.back().frameIndex;
                    if (i > lastIdx && (i - lastIdx) < minPeakFrames) {
                        if (odf[i] > candidates.back().strength) {
                            // Replace the previous (we keep only the stronger peak within the window)
                            candidates.pop_back();
                        } else {
                            suppressed = true;
                        }
                    }
                }
                if (!suppressed) {
                    double time = static_cast<double>(i) / frameRate;
                    candidates.emplace_back(time, odf[i], i);
                }
            }
        }
    }

    return candidates;
}

// Compute tempogram - tempo salience over time
std::vector<std::vector<double>> RealBPMModule::computeTempogram(const std::vector<double>& odf, double frameRate) {
    const double minBPM = static_cast<double>(m_cfg.minBPM);
    const double maxBPM = static_cast<double>(m_cfg.maxBPM);
    const int tempoBins = 120; // Resolution of tempo analysis
    const double tempogramWindowSec = 4.0; // Window size for local tempo analysis

    size_t windowFrames = static_cast<size_t>(tempogramWindowSec * frameRate);
    size_t hopFrames = std::max<size_t>(1, windowFrames / 4); // 75% overlap

    std::vector<std::vector<double>> tempogram;

    // Generate tempo candidates (logarithmic spacing)
    std::vector<double> tempoCandidates(tempoBins);
    double logMinBPM = std::log(minBPM);
    double logMaxBPM = std::log(maxBPM);
    for (int i = 0; i < tempoBins; ++i) {
        double logBPM = logMinBPM + (logMaxBPM - logMinBPM) * i / (tempoBins - 1);
        tempoCandidates[i] = std::exp(logBPM);
    }

    // Sliding window analysis
    for (size_t start = 0; start + windowFrames <= odf.size(); start += hopFrames) {
        std::vector<double> tempoSalience(tempoBins, 0.0);

        // Extract window
        std::vector<double> window(odf.begin() + start, odf.begin() + start + windowFrames);

        // Precompute window energy for comb normalization
        double windowEnergy = 0.0;
        for (double v : window) windowEnergy += v * v;
        if (windowEnergy < 1e-12) windowEnergy = 1e-12;

        // For each tempo candidate, compute autocorrelation strength
        for (int t = 0; t < tempoBins; ++t) {
            double bpm = tempoCandidates[t];
            double period = 60.0 / bpm; // Period in seconds
            int lag = static_cast<int>(period * frameRate); // Period in frames

            if (lag >= 2 && lag < static_cast<int>(windowFrames) - 1) {
                // Compute autocorrelation at this lag
                double autocorr = 0.0;
                double norm = 0.0;

                for (size_t i = 0; i < windowFrames - lag; ++i) {
                    autocorr += window[i] * window[i + lag];
                    norm += window[i] * window[i];
                }

                if (norm > 1e-10) {
                    tempoSalience[t] = autocorr / norm;
                }

                // Also check harmonic relationships (double and half tempo)
                int lag2 = lag / 2;
                int lag3 = lag * 2;

                if (lag2 >= 2 && lag2 < static_cast<int>(windowFrames) - 1) {
                    double autocorr2 = 0.0;
                    for (size_t i = 0; i < windowFrames - lag2; ++i) {
                        autocorr2 += window[i] * window[i + lag2];
                    }
                    if (norm > 1e-10) {
                        tempoSalience[t] += 0.3 * (autocorr2 / norm); // Weaker contribution
                    }
                }

                if (lag3 < static_cast<int>(windowFrames) - 1) {
                    double autocorr3 = 0.0;
                    for (size_t i = 0; i < windowFrames - lag3; ++i) {
                        autocorr3 += window[i] * window[i + lag3];
                    }
                    if (norm > 1e-10) {
                        tempoSalience[t] += 0.3 * (autocorr3 / norm); // Weaker contribution
                    }
                }

                // R4: Hybrid tempogram comb-like blending
                if (m_hybridTempogram) {
                    int Hcomb = std::max(2, m_combHarmonics);
                    double combSum = 0.0;
                    double wsum = 0.0;
                    for (int h = 1; h <= Hcomb; ++h) {
                        size_t lagH = static_cast<size_t>(h * lag);
                        if (lagH < static_cast<size_t>(windowFrames) - 1) {
                            double acch = 0.0;
                            for (size_t i = 0; i < windowFrames - lagH; ++i) {
                                acch += window[i] * window[i + lagH];
                            }
                            double w = 1.0 / static_cast<double>(h * h);
                            double corrh = acch / windowEnergy;
                            combSum += w * corrh;
                            wsum += w;
                        }
                    }
                    if (wsum > 0.0) {
                        double comb = combSum / wsum;
                        double lam = std::max(0.0, std::min(1.0, m_combLambda));
                        tempoSalience[t] = (1.0 - lam) * tempoSalience[t] + lam * comb;
                    }
                }
            }
        }

        // Normalize and smooth the tempo salience
        double maxSalience = *std::max_element(tempoSalience.begin(), tempoSalience.end());
        if (maxSalience > 1e-10) {
            for (double& s : tempoSalience) {
                s /= maxSalience;
            }
        }

        tempogram.push_back(tempoSalience);
    }

    return tempogram;
}

// Dynamic Programming Beat Tracking (Viterbi-style algorithm)
std::vector<double> RealBPMModule::trackBeatsWithDynamicProgramming(
    const std::vector<BeatCandidate>& candidates,
    const std::vector<std::vector<double>>& tempogram,
    double frameRate, double duration) {

    if (candidates.empty()) return {};

    const size_t N = candidates.size();
    const double minBPM = static_cast<double>(m_cfg.minBPM);
    const double maxBPM = static_cast<double>(m_cfg.maxBPM);

    // Dynamic programming tables
    std::vector<double> score(N, -1e9);           // Best score to reach each candidate
    std::vector<int> predecessor(N, -1);          // Best predecessor for backtracking

    // Initialize first few candidates
    for (size_t i = 0; i < std::min<size_t>(N, 10); ++i) {
        score[i] = candidates[i].strength; // Initial score = onset strength
    }

    // Forward pass: fill DP table
    for (size_t i = 1; i < N; ++i) {
        const auto& current = candidates[i];

        // Look back for potential predecessors
        for (size_t j = 0; j < i; ++j) {
            const auto& prev = candidates[j];

            double interval = current.time - prev.time;
            if (interval <= 0.0) continue;

            double impliedBPM = 60.0 / interval;

            // Check if implied BPM is reasonable
            if (impliedBPM < minBPM || impliedBPM > maxBPM) continue;

            // Compute transition cost
            double transitionCost = computeTransitionCost(prev, current, tempogram, frameRate);

            // Compute new score
            double newScore = score[j] + current.strength + transitionCost;

            if (newScore > score[i]) {
                score[i] = newScore;
                predecessor[i] = static_cast<int>(j);
            }
        }
    }

    // Find the best ending point
    int bestEnd = -1;
    double bestScore = -1e9;
    for (size_t i = N - std::min<size_t>(N, 10); i < N; ++i) {
        if (score[i] > bestScore) {
            bestScore = score[i];
            bestEnd = static_cast<int>(i);
        }
    }

    if (bestEnd == -1) return {};

    // Backward pass: reconstruct the best path
    std::vector<double> beatTimes;
    int current = bestEnd;
    while (current != -1) {
        beatTimes.push_back(candidates[current].time);
        current = predecessor[current];
    }

    // Reverse to get chronological order
    std::reverse(beatTimes.begin(), beatTimes.end());

    // Fill in missing beats using median interval
    if (beatTimes.size() >= 2) {
        fillMissingBeats(beatTimes, duration);
    }

    // Octave correction post-processing
    if (m_octaveCorrection) {
        beatTimes = postProcessOctaveCorrection(beatTimes, duration, tempogram);
    }

    return beatTimes;
}

// Helper method to compute transition cost between two beat candidates
double RealBPMModule::computeTransitionCost(const BeatCandidate& prev, const BeatCandidate& current,
                                          const std::vector<std::vector<double>>& tempogram, double frameRate) {
    double interval = current.time - prev.time;
    if (interval <= 0.0) return -1e9;
    double impliedBPM = 60.0 / interval;

    // Base cost: prefer intervals that match tempogram
    double tempogramSupport = 0.0;
    double harmonicBonus = 0.0;
    double tempoChangePenalty = 0.0;

    // Find tempogram frame for current beat
    const double tempogramWindowSec = 4.0;
    const size_t tempogramHop = static_cast<size_t>(tempogramWindowSec * frameRate / 4);
    const size_t frameIndex = static_cast<size_t>(current.time * frameRate);
    size_t tempogramIndex = frameIndex / tempogramHop;

    const double minBPM = static_cast<double>(m_cfg.minBPM);
    const double maxBPM = static_cast<double>(m_cfg.maxBPM);

    if (!tempogram.empty()) {
        if (tempogramIndex >= tempogram.size()) {
            tempogramIndex = tempogram.size() - 1;
        }
        const int tempoBins = static_cast<int>(tempogram[0].size());
        const double logMinBPM = std::log(minBPM);
        const double logMaxBPM = std::log(maxBPM);

        // Support at implied BPM bin
        double logImpliedBPM = std::log(impliedBPM);
        double binFloat = (logImpliedBPM - logMinBPM) / (logMaxBPM - logMinBPM) * (tempoBins - 1);
        int bin = std::max(0, std::min(tempoBins - 1, static_cast<int>(binFloat)));
        tempogramSupport = tempogram[tempogramIndex][bin];

        // Find dominant local BPM from tempogram at this time
        const auto& row = tempogram[tempogramIndex];
        int domBin = static_cast<int>(std::max_element(row.begin(), row.end()) - row.begin());
        double domLogBPM = logMinBPM + (logMaxBPM - logMinBPM) * (static_cast<double>(domBin) / (tempoBins - 1));
        double dominantBPM = std::exp(domLogBPM);

        // Harmonic relationship bonus between implied and dominant tempo
        if (dominantBPM > 0.0) {
            double ratio = impliedBPM / dominantBPM;
            double tolStrong = 0.08;  // ~8% tolerance
            double tolWeak = 0.06;
            auto near = [](double x, double t, double tol){ return std::abs(x - t) < tol; };
            if (near(ratio, 1.0, tolStrong)) {
                harmonicBonus += 0.25;
            } else if (near(ratio, 2.0, tolStrong) || near(ratio, 0.5, tolStrong)) {
                harmonicBonus += 0.15;
            } else if (near(ratio, 1.5, tolWeak) || near(ratio, 2.0/3.0, tolWeak)) {
                harmonicBonus += 0.10;
            }

            // Tempo change penalty if deviating strongly from dominant tempo
            if (m_fixedTempo) {
                if (ratio < 0.90 || ratio > 1.10) {
                    tempoChangePenalty = 0.30;
                } else if (ratio < 0.95 || ratio > 1.05) {
                    tempoChangePenalty = 0.15;
                }
            } else {
                if (ratio < 0.67 || ratio > 1.5) {
                    tempoChangePenalty = 0.20;
                } else if (ratio < 0.8 || ratio > 1.25) {
                    tempoChangePenalty = 0.10;
                }
            }
        }
    }

    // Musical preference: favor common BPM ranges (reduced to avoid bias)
    double musicalBonus = 0.0;
    if (impliedBPM >= 100.0 && impliedBPM <= 140.0) {
        musicalBonus = 0.30;  // Reduced bias vs previous 0.5
    } else if (impliedBPM >= 80.0 && impliedBPM <= 180.0) {
        musicalBonus = 0.15;  // Reduced bias vs previous 0.2
    }

    // Combined transition cost (higher is better)
    return tempogramSupport + harmonicBonus + musicalBonus - tempoChangePenalty;
}

// Helper method to fill missing beats using interpolation
void RealBPMModule::fillMissingBeats(std::vector<double>& beatTimes, double duration) {
    if (beatTimes.size() < 2) return;

    // Compute intervals
    std::vector<double> intervals;
    for (size_t i = 1; i < beatTimes.size(); ++i) {
        intervals.push_back(beatTimes[i] - beatTimes[i-1]);
    }

    // Compute median interval
    std::sort(intervals.begin(), intervals.end());
    double medianInterval = intervals[intervals.size() / 2];

    // Fill backward from first beat
    double time = beatTimes[0] - medianInterval;
    std::vector<double> prependBeats;
    while (time > 0.0) {
        prependBeats.push_back(time);
        time -= medianInterval;
    }
    std::reverse(prependBeats.begin(), prependBeats.end());

    // Fill forward from last beat
    std::vector<double> appendBeats;
    time = beatTimes.back() + medianInterval;
    while (time < duration) {
        appendBeats.push_back(time);
        time += medianInterval;
    }

    // Combine all beats
    std::vector<double> allBeats;
    allBeats.insert(allBeats.end(), prependBeats.begin(), prependBeats.end());
    allBeats.insert(allBeats.end(), beatTimes.begin(), beatTimes.end());
    allBeats.insert(allBeats.end(), appendBeats.begin(), appendBeats.end());

    beatTimes = allBeats;
}

// Post-processing: Octave correction (0.5x/1x/2x)
std::vector<double> RealBPMModule::postProcessOctaveCorrection(const std::vector<double>& beatTimes,
                                                              double duration,
                                                              const std::vector<std::vector<double>>& tempogram) {
    if (beatTimes.size() < 2) return beatTimes;

    // Compute base interval and BPMs
    std::vector<double> intervals;
    intervals.reserve(beatTimes.size() - 1);
    for (size_t i = 1; i < beatTimes.size(); ++i) {
        double d = beatTimes[i] - beatTimes[i-1];
        if (d > 0.0) intervals.push_back(d);
    }
    if (intervals.empty()) return beatTimes;
    std::sort(intervals.begin(), intervals.end());
    double medianInterval = intervals[intervals.size() / 2];
    double bpm1x = 60.0 / medianInterval;
    double bpm05 = bpm1x * 0.5;
    double bpm2x = bpm1x * 2.0;

    auto clamp01 = [](double x){ return std::max(0.0, std::min(1.0, x)); };

    auto makeGrid = [&](double bpm){
        double interval = 60.0 / bpm;
        // Anchor grid to first detected beat
        double start = beatTimes.front();
        // backfill to zero to cover intro
        while (start - interval > 0.0) start -= interval;
        std::vector<double> grid;
        for (double t = start; t <= duration + 1e-6; t += interval) grid.push_back(t);
        return grid;
    };

    auto tempogramEvidence = [&](double bpm){
        if (tempogram.empty()) return 0.0;
        const double minBPM = static_cast<double>(m_cfg.minBPM);
        const double maxBPM = static_cast<double>(m_cfg.maxBPM);
        const int tempoBins = static_cast<int>(tempogram[0].size());
        if (bpm < minBPM || bpm > maxBPM || tempoBins <= 0) return 0.0;
        double logMinBPM = std::log(minBPM);
        double logMaxBPM = std::log(maxBPM);
        double logBPM = std::log(bpm);
        double binFloat = (logBPM - logMinBPM) / (logMaxBPM - logMinBPM) * (tempoBins - 1);
        int bin = std::max(0, std::min(tempoBins - 1, static_cast<int>(binFloat)));
        double sum = 0.0;
        for (const auto& row : tempogram) sum += row[bin];
        // Fix: Removed check tempogram.size() > 0 (always true here)
        return sum / static_cast<double>(tempogram.size());
    };

    auto coverageMatch = [&](const std::vector<double>& grid){
        if (grid.empty()) return 0.0;
        double interval = (grid.size() >= 2) ? (grid[1] - grid[0]) : medianInterval;
        double tol = 0.1 * interval;
        // fraction of detected beats close to grid
        size_t i = 0, j = 0; size_t closeDet = 0;
        while (i < beatTimes.size() && j < grid.size()) {
            double diff = beatTimes[i] - grid[j];
            if (std::abs(diff) <= tol) { ++closeDet; ++i; ++j; }
            else if (diff > 0) { ++j; } else { ++i; }
        }
        double fracDet = static_cast<double>(closeDet) / static_cast<double>(beatTimes.size());
        // fraction of grid beats close to detected beats
        i = 0; j = 0; size_t closeGrid = 0;
        while (i < beatTimes.size() && j < grid.size()) {
            double diff = grid[j] - beatTimes[i];
            if (std::abs(diff) <= tol) { ++closeGrid; ++i; ++j; }
            else if (diff > 0) { ++i; } else { ++j; }
        }
        double fracGrid = static_cast<double>(closeGrid) / static_cast<double>(grid.size());
        // Use conservative matching: take the minimum, not the average
        return std::min(fracDet, fracGrid);
    };

    // Precompute global interval stability from detected beats (candidate-independent)
    double meanInt = 0.0; for (double v : intervals) meanInt += v; meanInt /= intervals.size();
    double varInt = 0.0; for (double v : intervals) { double d = v - meanInt; varInt += d*d; }
    varInt /= intervals.size();
    double stdInt = std::sqrt(varInt);
    const double stabilityFixed = clamp01(1.0 - (stdInt / std::max(1e-9, meanInt)));

    struct Cand { double bpm; std::vector<double> grid; double score; double cov; double ev; };
    std::vector<Cand> cands;
    cands.reserve(3);
    for (double b : {bpm05, bpm1x, bpm2x}) {
        Cand c; c.bpm = b; c.grid = makeGrid(b);
        c.ev = tempogramEvidence(b);
        c.cov = coverageMatch(c.grid);
        const double alpha = 0.55, beta = 0.35, gamma = 0.10; // emphasize tempogram and coverage
        c.score = alpha * c.ev + beta * c.cov + gamma * stabilityFixed;
        cands.push_back(std::move(c));
    }

    // Find best and compare vs 1x
    int idx1x = 1; // order is {0.5x, 1x, 2x}
    int bestIdx = idx1x;
    for (int i = 0; i < static_cast<int>(cands.size()); ++i) {
        if (cands[i].score > cands[bestIdx].score) bestIdx = i;
    }

    // Require strong improvement and better or comparable coverage to switch octave
    double relGain = (cands[bestIdx].score - cands[idx1x].score) / std::max(1e-6, cands[idx1x].score);
    bool coverageGuard = (cands[idx1x].cov >= 0.85) && (cands[bestIdx].cov < cands[idx1x].cov + 0.05);
    bool switchOctave = (bestIdx != idx1x && relGain > 0.15 && !coverageGuard);
    m_octaveSwitchedLast = switchOctave;
    if (switchOctave) {
        return cands[bestIdx].grid; // adopt corrected octave grid
    }
    return beatTimes; // keep original
}

// Generate final beat tracking result in expected JSON format
nlohmann::json RealBPMModule::generateBeatTrackingResult(const std::vector<double>& beatTimes, double duration) {
    if (beatTimes.size() < 2) {
        return makeResultFallback(duration);
    }

    // Compute intervals and median interval
    std::vector<double> intervals;
    intervals.reserve(beatTimes.size() - 1);
    for (size_t i = 1; i < beatTimes.size(); ++i) {
        double d = beatTimes[i] - beatTimes[i-1];
        if (d > 0.0) intervals.push_back(d);
    }
    if (intervals.empty()) return makeResultFallback(duration);

    std::sort(intervals.begin(), intervals.end());
    const double medianInterval = intervals[intervals.size() / 2];

    // Refine BPM using least-squares fit of beat index vs time (global period)
    double periodLS = medianInterval;
    {
        const size_t M = beatTimes.size();
        if (M >= 3) {
            // Use middle 80% of beats to reduce boundary effects
            size_t i0 = static_cast<size_t>(M * 0.1);
            size_t i1 = static_cast<size_t>(M * 0.9);
            if (i1 <= i0 + 1) { i0 = 0; i1 = M; }
            const size_t K = i1 - i0;
            if (K >= 2) {
                double sumK = 0.0, sumT = 0.0, sumKK = 0.0, sumKT = 0.0;
                for (size_t k = 0; k < K; ++k) {
                    double ki = static_cast<double>(i0 + k);
                    double ti = beatTimes[i0 + k];
                    sumK += ki; sumT += ti; sumKK += ki * ki; sumKT += ki * ti;
                }
                double denom = (K * sumKK - sumK * sumK);
                if (std::abs(denom) > 1e-12) {
                    double slope = (K * sumKT - sumK * sumT) / denom; // seconds per beat index
                    if (slope > 1e-6 && std::isfinite(slope)) {
                        periodLS = slope;
                    }
                }
            }
        }
    }
    // Refine global period by maximizing circular alignment around LS estimate
    double periodRef = periodLS;
    if (periodLS > 0.0 && beatTimes.size() >= 8) {
        const double PI = 3.14159265358979323846;
        const double maxRel = 0.01; // ±1%
        const int steps = 81;       // ~0.025% per step
        double bestR = -1.0;
        double bestP = periodLS;
        for (int s = 0; s < steps; ++s) {
            double rel = -maxRel + (2.0 * maxRel) * (static_cast<double>(s) / (steps - 1));
            double P = periodLS * (1.0 + rel);
            if (P <= 1e-6) continue;
            double omega = 2.0 * PI / P;
            double sx = 0.0, sy = 0.0;
            for (double t : beatTimes) {
                double ang = omega * t;
                sx += std::cos(ang);
                sy += std::sin(ang);
            }
            double R = std::sqrt(sx*sx + sy*sy) / static_cast<double>(beatTimes.size());
            if (R > bestR) { bestR = R; bestP = P; }
        }
        if (bestR > 0.0) {
            periodRef = bestP;
        }
    }
    double bpm = 60.0 / periodRef;

    // Global interval consistency (kept as principal factor)
    double mean = 0.0;
    for (double v : intervals) mean += v;
    mean /= static_cast<double>(intervals.size());
    double var = 0.0;
    for (double v : intervals) { double d = v - mean; var += d * d; }
    var /= static_cast<double>(intervals.size());
    const double intervalStdDev = std::sqrt(var);
    const double globalConsistency = std::max(0.0, std::min(1.0, 1.0 - (intervalStdDev / medianInterval) * 2.0));

    // Coverage: how many beats we have vs. how many expected
    const double expectedBeats = std::max(1.0, duration / periodRef);
    const double rawCoverage = static_cast<double>(beatTimes.size()) / expectedBeats;
    const double coverageScore = std::max(0.0, std::min(1.0, rawCoverage));

    // Local stability: sliding 4-second window std-dev of local intervals (lower is better)
    const double windowSec = 4.0;
    std::vector<double> localStdNorms;
    size_t startIdx = 0;
    for (size_t i = 0; i < beatTimes.size(); ++i) {
        while (startIdx < i && (beatTimes[i] - beatTimes[startIdx]) > windowSec) {
            ++startIdx;
        }
        if (i > startIdx + 1) {
            // collect intervals within [startIdx, i]
            std::vector<double> winIntervals;
            winIntervals.reserve(i - startIdx);
            for (size_t k = startIdx + 1; k <= i; ++k) {
                double d = beatTimes[k] - beatTimes[k-1];
                if (d > 0.0) winIntervals.push_back(d);
            }
            if (winIntervals.size() >= 2) {
                double m = 0.0; for (double v : winIntervals) m += v; m /= winIntervals.size();
                double vv = 0.0; for (double v : winIntervals) { double dd = v - m; vv += dd * dd; }
                vv /= winIntervals.size();
                double s = std::sqrt(vv);
                localStdNorms.push_back(s / medianInterval);
            }
        }
    }
    double localInstability = 0.0;
    if (!localStdNorms.empty()) {
        for (double x : localStdNorms) localInstability += x;
        localInstability /= static_cast<double>(localStdNorms.size());
    }
    const double localStabilityScore = std::max(0.0, std::min(1.0, 1.0 - localInstability));

    // Composite confidence: weights w1=0.6 (global), w3=0.2 (coverage), w4=0.2 (local stability)
    const double w1 = 0.6, w3 = 0.2, w4 = 0.2;
    double confidence = w1 * globalConsistency + w3 * coverageScore + w4 * localStabilityScore;
    confidence = std::max(0.0, std::min(1.0, confidence));

    // Create beat grid JSON
    nlohmann::json beatGrid = nlohmann::json::array();
    for (double t : beatTimes) {
        if (t >= 0.0 && t <= duration) {
            beatGrid.push_back({{"t", static_cast<float>(t)}, {"strength", 1.0f}});
        }
    }

    // Downbeats (every 4th beat)
    nlohmann::json downbeats = nlohmann::json::array();
    for (size_t i = 0; i < beatGrid.size(); i += 4) {
        downbeats.push_back(beatGrid[i]["t"]);
    }

    // Annotate method/engine based on active configuration
    std::string method;
    const char* engine = nullptr;
    if (m_useQMDSP) {
        method = "qm-dsp";
        engine = "qm-dsp";
    } else {
        method = "beat-tracking-dp";
        if (m_qmLike) method = "qm-like-dp";
        if (m_hybridTempogram) method += "+hybrid";
        if (m_fixedTempo) method += "+fixed";
        engine = m_qmLike ? "qm-like" : "native";
    }

    nlohmann::json j = {{"bpm", bpm},
            {"confidence", confidence},
            {"beatInterval", static_cast<float>(medianInterval)},
            {"beatGrid", beatGrid},
            {"downbeats", downbeats},
            {"method", method},
            {"engine", engine}};

    // C2: Health metrics and flags
    // Fraction of intervals near octave-related durations (0.5x or 2x of median)
    size_t altCount = 0;
    for (double v : intervals) {
        double rHalf = std::abs(v - 0.5 * medianInterval) / std::max(1e-9, 0.5 * medianInterval);
        double rDouble = std::abs(v - 2.0 * medianInterval) / std::max(1e-9, 2.0 * medianInterval);
        if (std::min(rHalf, rDouble) < 0.08) {
            ++altCount;
        }
    }
    const double altHarmonicFraction = static_cast<double>(altCount) / intervals.size();

    const bool tempoDriftSuspected = (localInstability > 0.18) || (globalConsistency < 0.65);
    const bool octaveAmbiguityHigh = (altHarmonicFraction > 0.30);
    const bool lowCoverage = (rawCoverage < 0.85);
    const bool analysisTruncated = (m_fastAnalysisSec > 0.0 && duration > m_fastAnalysisSec + 0.5);

    j["health"] = {
        {"tempoDriftSuspected", tempoDriftSuspected},
        {"octaveAmbiguityHigh", octaveAmbiguityHigh},
        {"lowCoverage", lowCoverage},
        {"analysisTruncated", analysisTruncated},
        {"fixedTempoAssumption", m_fixedTempo},
        {"octaveCorrectionApplied", m_octaveSwitchedLast}
    };

    j["metrics"] = {
        {"intervalStdRel", intervalStdDev / std::max(1e-9, medianInterval)},
        {"coverage", rawCoverage},
        {"localInstability", localInstability},
        {"altHarmonicFraction", altHarmonicFraction}
    };

    return j;
}

std::unique_ptr<core::IAnalysisModule> createRealBPMModule() { return std::make_unique<RealBPMModule>(); }

} // namespace ave::modules

// QM-DSP direct engine processing using DetectionFunction + TempoTrackV2
std::vector<double> ave::modules::RealBPMModule::processWithQMDSP(const std::vector<float>& mono, float sr, double duration) {
    // Reset last ODF state
    m_lastODF.clear();
    m_lastODFFrameRate = 0.0;
    // Parameters aligned with Mixxx AnalyzerQueenMaryBeats
    constexpr float kStepSecs = 0.01161f;

    // Use Mixxx-identical integer math for step and window sizes
    const int sampleRate = static_cast<int>(std::lround(sr));
    const int stepSize = std::max(1, static_cast<int>(sampleRate * kStepSecs)); // truncation like Mixxx
    int windowSize = MathUtilities::nextPowerOfTwo(sampleRate / 50); // integer division like Mixxx
    if (windowSize < 256) windowSize = 256;

    DFConfig cfg{};
    cfg.DFType = DF_COMPLEXSD;
    cfg.stepSize = stepSize;
    cfg.frameLength = windowSize;
    cfg.dbRise = 3;
    cfg.adaptiveWhitening = false;
    cfg.whiteningRelaxCoeff = -1;
    cfg.whiteningFloor = -1;

    DetectionFunction det(cfg);

    if (mono.size() < static_cast<size_t>(windowSize)) return {};

    std::vector<double> detectionResults;
    detectionResults.reserve(1 + (mono.size() - static_cast<size_t>(windowSize)) / static_cast<size_t>(stepSize));
    std::vector<double> frame(windowSize, 0.0);

    const size_t total = mono.size();
    for (size_t start = 0; start + static_cast<size_t>(windowSize) <= total; start += static_cast<size_t>(stepSize)) {
        // copy mono to double buffer (no windowing: qm-dsp does its own)
        for (int i = 0; i < windowSize; ++i) frame[i] = static_cast<double>(mono[start + i]);
        double val = det.processTimeDomain(frame.data());
        detectionResults.push_back(val);
    }

    int nonZeroCount = static_cast<int>(detectionResults.size());
    while (nonZeroCount > 0 && detectionResults[nonZeroCount - 1] <= 0.0) {
        --nonZeroCount;
    }
    if (nonZeroCount <= 2) return {};

    std::vector<double> df;
    std::vector<double> beatPeriod;
    df.reserve(nonZeroCount - 2);
    beatPeriod.reserve(nonZeroCount - 2);

    for (int i = 2; i < nonZeroCount; ++i) {
        df.push_back(detectionResults[i]);
        beatPeriod.push_back(0.0);
    }

    // Store ODF for downstream consumers
    m_lastODF = df;
    m_lastODFFrameRate = (stepSize > 0) ? (static_cast<double>(sampleRate) / static_cast<double>(stepSize)) : 0.0;

    TempoTrackV2 tt(static_cast<float>(sampleRate), stepSize);
    tt.calculateBeatPeriod(df, beatPeriod);

    std::vector<double> beats;
    tt.calculateBeats(df, beatPeriod, beats);

    std::vector<double> beatTimes;
    beatTimes.reserve(beats.size());
    for (size_t i = 0; i < beats.size(); ++i) {
        // Convert beat index to sample position and then to seconds (match Mixxx + half-hop offset)
        double samplePos = (beats[i] * stepSize) + (stepSize / 2.0);
        double t = samplePos / static_cast<double>(sampleRate);
        if (t >= 0.0 && t <= duration + 1e-6) {
            beatTimes.push_back(t);
        }
    }

    // Ensure sorted & deduplicated
    std::sort(beatTimes.begin(), beatTimes.end());
    beatTimes.erase(std::unique(beatTimes.begin(), beatTimes.end(), [](double a, double b){ return std::abs(a-b) < 1e-6; }), beatTimes.end());

    return beatTimes;
}