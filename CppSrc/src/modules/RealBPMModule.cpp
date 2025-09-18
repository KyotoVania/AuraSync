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
        return generateBeatTrackingResult(beatTimes, audio.getDuration());
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
    std::vector<float> m_history;
    
    // New beat tracking methods
    std::vector<double> extractComplexSpectralDifferenceODF(const std::vector<float>& mono, size_t N, size_t H, float sr);
    std::vector<BeatCandidate> detectBeatCandidates(const std::vector<double>& odf, double frameRate);
    std::vector<std::vector<double>> computeTempogram(const std::vector<double>& odf, double frameRate);
    std::vector<double> trackBeatsWithDynamicProgramming(const std::vector<BeatCandidate>& candidates, 
                                                        const std::vector<std::vector<double>>& tempogram, 
                                                        double frameRate, double duration);
    nlohmann::json generateBeatTrackingResult(const std::vector<double>& beatTimes, double duration);
    
    // Helper methods for dynamic programming
    double computeTransitionCost(const BeatCandidate& prev, const BeatCandidate& current, 
                               const std::vector<std::vector<double>>& tempogram, double frameRate);
    void fillMissingBeats(std::vector<double>& beatTimes, double duration);

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
    
    // FFTW buffers
    double* in = (double*)fftw_malloc(sizeof(double) * N);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
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
            if (ratio < 0.67 || ratio > 1.5) {
                tempoChangePenalty = 0.20;
            } else if (ratio < 0.8 || ratio > 1.25) {
                tempoChangePenalty = 0.10;
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

// Generate final beat tracking result in expected JSON format
nlohmann::json RealBPMModule::generateBeatTrackingResult(const std::vector<double>& beatTimes, double duration) {
    if (beatTimes.size() < 2) {
        return makeResultFallback(duration);
    }
    
    // Compute BPM from median interval
    std::vector<double> intervals;
    for (size_t i = 1; i < beatTimes.size(); ++i) {
        intervals.push_back(beatTimes[i] - beatTimes[i-1]);
    }
    
    std::sort(intervals.begin(), intervals.end());
    double medianInterval = intervals[intervals.size() / 2];
    double bpm = 60.0 / medianInterval;
    
    // Estimate confidence based on interval consistency
    double intervalVariance = 0.0;
    for (double interval : intervals) {
        double diff = interval - medianInterval;
        intervalVariance += diff * diff;
    }
    intervalVariance /= intervals.size();
    double intervalStdDev = std::sqrt(intervalVariance);
    
    // Confidence: higher when intervals are more consistent
    double confidence = std::max(0.0, std::min(1.0, 1.0 - (intervalStdDev / medianInterval) * 2.0));
    
    // Create beat grid
    nlohmann::json beatGrid = nlohmann::json::array();
    for (double t : beatTimes) {
        if (t >= 0.0 && t <= duration) {
            beatGrid.push_back({
                {"t", static_cast<float>(t)}, 
                {"strength", 1.0f}
            });
        }
    }
    
    // Create downbeats (every 4th beat)
    nlohmann::json downbeats = nlohmann::json::array();
    for (size_t i = 0; i < beatGrid.size(); i += 4) {
        downbeats.push_back(beatGrid[i]["t"]);
    }
    
    // Return result with new method identifier
    return {
        {"bpm", bpm},
        {"confidence", confidence},
        {"beatInterval", static_cast<float>(medianInterval)},
        {"beatGrid", beatGrid},
        {"downbeats", downbeats},
        {"method", "beat-tracking-dp"}  // Identify the new approach
    };
}

std::unique_ptr<core::IAnalysisModule> createRealBPMModule() { return std::make_unique<RealBPMModule>(); }

} // namespace ave::modules
