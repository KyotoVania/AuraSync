#include "../../include/core/IAnalysisModule.h"
#include "../../include/core/AudioBuffer.h"
#include "../../include/modules/TonalityModule.h"
#include <nlohmann/json.hpp>
#include <fftw3.h>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <map>
#include <memory>

// Queen Mary DSP
#include <dsp/keydetection/GetKeyMode.h>

namespace ave { namespace modules {

// Lightweight helper inspired by Mixxx DownmixAndOverlapHelper
class DownmixAndOverlapHelper {
public:
    using WindowReadyCallback = std::function<bool(double* pBuffer, size_t currentFrame)>;

    bool initialize(size_t windowSize, size_t stepSize, const WindowReadyCallback& callback) {
        m_buffer.assign(windowSize, 0.0);
        m_windowSize = windowSize;
        m_stepSize = stepSize;
        m_callback = callback;
        m_writePos = windowSize / 2; // center first frame
        m_totalFramesProcessed = 0;
        return m_windowSize > 0 && m_stepSize > 0 && m_stepSize <= m_windowSize && static_cast<bool>(m_callback);
    }

    bool processStereoSamples(const float* pInputStereo, size_t inputStereoSamples) {
        const size_t numInputFrames = inputStereoSamples / 2;
        return processInner(pInputStereo, numInputFrames);
    }

    bool finalize() {
        // Pad with silence to flush last window
        size_t framesToFillWindow = (m_windowSize > m_writePos) ? (m_windowSize - m_writePos) : 0;
        size_t numInputFrames = std::max(framesToFillWindow, m_windowSize / 2 - 1);
        return processInner(nullptr, numInputFrames);
    }

private:
    bool processInner(const float* pInputStereo, size_t numInputFrames) {
        size_t inRead = 0;
        double* pDownmix = m_buffer.data();

        while (inRead < numInputFrames) {
            size_t readAvailable = numInputFrames - inRead;
            size_t writeAvailable = (m_writePos <= m_windowSize) ? (m_windowSize - m_writePos) : 0;
            size_t numFrames = std::min(readAvailable, writeAvailable);
            if (pInputStereo) {
                for (size_t i = 0; i < numFrames; ++i) {
                    // downmix stereo to mono (average L/R)
                    pDownmix[m_writePos + i] = (static_cast<double>(pInputStereo[(inRead + i) * 2]) +
                                                static_cast<double>(pInputStereo[(inRead + i) * 2 + 1])) * 0.5;
                }
            } else {
                for (size_t i = 0; i < numFrames; ++i) {
                    pDownmix[m_writePos + i] = 0.0;
                }
            }
            m_writePos += numFrames;
            inRead += numFrames;
            m_totalFramesProcessed += numFrames;

            if (m_writePos == m_windowSize) {
                bool ok = m_callback(pDownmix, m_totalFramesProcessed);
                if (!ok) return false;
                // overlap: shift by stepSize
                const size_t overlap = m_windowSize - m_stepSize;
                for (size_t i = 0; i < overlap; ++i) {
                    pDownmix[i] = pDownmix[i + m_stepSize];
                }
                m_writePos -= m_stepSize;
            }
        }
        return true;
    }

    std::vector<double> m_buffer;
    size_t m_windowSize = 0;
    size_t m_stepSize = 0;
    size_t m_writePos = 0;
    size_t m_totalFramesProcessed = 0;
    WindowReadyCallback m_callback;
};

class RealTonalityModule : public core::IAnalysisModule {
public:
    std::string getName() const override { return "Tonality"; }
    std::string getVersion() const override { return "1.1.0"; }
    bool isRealTime() const override { return true; }

    bool initialize(const nlohmann::json& config) override {
        // Keep legacy chroma config for test compatibility
        if (config.contains("fftSize")) m_fftSize = std::max<size_t>(256, config["fftSize"].get<size_t>());
        if (config.contains("hopSize")) m_hopSize = std::max<size_t>(1, config["hopSize"].get<size_t>());
        if (m_hopSize > m_fftSize) m_hopSize = m_fftSize / 2; // clamp
        if (config.contains("windowType")) m_windowType = config["windowType"].get<std::string>();
        if (config.contains("referenceFreq")) m_refFreq = std::max(1e-3, config["referenceFreq"].get<double>());

        // Reset progressive analysis state
        m_prevKey = -1;
        m_resultKeys.clear();
        m_currentFrame = 0;
        m_pKeyMode.reset();
        m_helper = DownmixAndOverlapHelper{};
        return true;
    }

    void reset() override {
        m_prevKey = -1;
        m_resultKeys.clear();
        m_currentFrame = 0;
        m_pKeyMode.reset();
        m_helper = DownmixAndOverlapHelper{};
    }

    std::vector<std::string> getDependencies() const override { return {}; }

    nlohmann::json process(const core::AudioBuffer& audio, const core::AnalysisContext&) override {
        // Prepare mono signal (for chromaVector output compatibility)
        const float sr = audio.getSampleRate();
        std::vector<float> mono = audio.getMono();
        if (mono.empty()) {
            return makeEmptyResult();
        }

        // 1) Compute a simple global chroma vector for tests/compat
        std::vector<double> chroma = computeGlobalChroma(mono, sr);

        // 2) Progressive key detection using Queen Mary DSP
        if (!m_pKeyMode) {
            GetKeyMode::Config cfg(sr, static_cast<float>(m_refFreq));
            m_pKeyMode = std::make_unique<GetKeyMode>(cfg);
            size_t windowSize = static_cast<size_t>(m_pKeyMode->getBlockSize());
            size_t stepSize = static_cast<size_t>(m_pKeyMode->getHopSize());
            m_helper.initialize(windowSize, stepSize, [this](double* pWindow, size_t currentFrame) {
                int iKeyRaw = m_pKeyMode->process(pWindow);
                int idx24 = mapQMKeyToIndex24(iKeyRaw);
                if (idx24 < 0) {
                    return true; // continue, ignore invalid
                }
                if (idx24 != m_prevKey) {
                    m_resultKeys.push_back(std::make_pair(idx24, static_cast<double>(currentFrame)));
                    m_prevKey = idx24;
                }
                return true;
            });
        }

        // Feed audio as fake stereo to helper
        const size_t numInputFrames = mono.size();
        std::vector<float> stereo;
        stereo.resize(numInputFrames * 2);
        for (size_t i = 0; i < numInputFrames; ++i) {
            stereo[i * 2] = mono[i];
            stereo[i * 2 + 1] = mono[i];
        }
        m_currentFrame += numInputFrames;
        m_helper.processStereoSamples(stereo.data(), stereo.size());

        // Finalize progressive analysis and compute predominant key
        m_helper.finalize();
        m_pKeyMode.reset();

        if (m_resultKeys.empty()) {
            // Fallback to chroma-only heuristic if nothing detected
            return makeDefaultFromChroma(chroma);
        }

        // Durations per key index (0..23)
        std::map<int, double> keyDurations;
        for (size_t i = 0; i < m_resultKeys.size(); ++i) {
            int key = m_resultKeys[i].first;
            double start = m_resultKeys[i].second;
            double end = (i + 1 < m_resultKeys.size()) ? m_resultKeys[i + 1].second : static_cast<double>(m_currentFrame);
            keyDurations[key] += std::max(0.0, end - start);
        }
        int bestKey = -1;
        double maxDur = 0.0;
        for (const auto& kv : keyDurations) {
            if (kv.second > maxDur) { maxDur = kv.second; bestKey = kv.first; }
        }
        if (bestKey < 0) {
            return makeDefaultFromChroma(chroma);
        }
        const bool major = (bestKey < 12);
        const int tonicIndex = major ? bestKey : bestKey - 12;
        static const char* KEY_NAMES[12] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
        std::string keyName = KEY_NAMES[static_cast<size_t>(tonicIndex)];
        std::string mode = major ? "major" : "minor";
        std::string keyString = major ? keyName : (keyName + std::string("m"));
        double confidence = (m_currentFrame > 0) ? (maxDur / static_cast<double>(m_currentFrame)) : 0.0;
        confidence = std::clamp(confidence, 0.0, 1.0);

        nlohmann::json chromaJson = nlohmann::json::array();
        for (double v : chroma) chromaJson.push_back(v);

        // Compute per-frame chromagram sequence aligned with Spectral frames if available
        size_t Nseq = m_fftSize;
        size_t Hseq = m_hopSize == 0 ? std::max<size_t>(1, Nseq / 4) : m_hopSize;
        double frameRate = sr / static_cast<double>(Hseq);
        // Since we don't have Spectral config here, we use local FFT/hop; RealStructureModule will align by timestamps
        nlohmann::json chromaSeq = nlohmann::json::array();
        {
            // Window for sequence computation
            std::vector<float> winF;
            if (m_windowType == "hann" || m_windowType.empty()) winF = core::window::hann(Nseq);
            else if (m_windowType == "hamming") winF = core::window::hamming(Nseq);
            else if (m_windowType == "blackman") winF = core::window::blackman(Nseq);
            else winF = core::window::hann(Nseq);
            std::vector<double> window(winF.begin(), winF.end());

            size_t numFramesSeq = (mono.size() + Hseq - 1) / Hseq;
            double* inS = static_cast<double*>(fftw_malloc(sizeof(double) * Nseq));
            fftw_complex* outS = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (Nseq / 2 + 1)));
            fftw_plan planS = fftw_plan_dft_r2c_1d(static_cast<int>(Nseq), inS, outS, FFTW_ESTIMATE);
            const size_t numBinsS = Nseq / 2 + 1;

            for (size_t f = 0; f < numFramesSeq; ++f) {
                const size_t start = f * Hseq;
                for (size_t i = 0; i < Nseq; ++i) {
                    size_t idx = start + i;
                    double s = (idx < mono.size()) ? static_cast<double>(mono[idx]) : 0.0;
                    inS[i] = s * window[i];
                }
                fftw_execute(planS);
                std::vector<double> cv(12, 0.0);
                for (size_t k = 1; k < numBinsS; ++k) {
                    double re = outS[k][0];
                    double im = outS[k][1];
                    double power = re * re + im * im;
                    if (power <= 0.0) continue;
                    double fk = (static_cast<double>(k) * sr) / static_cast<double>(Nseq);
                    if (fk < 20.0) continue;
                    double pitch = 12.0 * std::log2(fk / m_refFreq) + 69.0;
                    long nearest = static_cast<long>(std::llround(pitch));
                    int cb = static_cast<int>(nearest % 12);
                    if (cb < 0) cb += 12;
                    cv[static_cast<size_t>(cb)] += power;
                }
                double ssum = 0.0; for (double v : cv) ssum += v;
                double ssum = std::accumulate(cv.begin(), cv.end(), 0.0);
                if (ssum > 0.0) {
                    std::transform(cv.begin(), cv.end(), cv.begin(), [ssum](double v) { return v / ssum; });
                }
                nlohmann::json vj = nlohmann::json::array();
                for (double x : cv) vj.push_back(x);
                chromaSeq.push_back({{"t", t}, {"v", vj}});
            }
            fftw_destroy_plan(planS);
            fftw_free(inS);
            fftw_free(outS);
        }

        nlohmann::json result = {
            {"key", keyName},
            {"mode", mode},
            {"keyString", keyString},
            {"confidence", confidence},
            {"chromaVector", chromaJson},
            {"chromaSequence", chromaSeq},
            {"frameRate", frameRate}
        };
        return result;
    }

    bool validateOutput(const nlohmann::json& output) const override {
        return output.contains("key") && output.contains("mode") && output.contains("confidence");
    }

private:
    // Legacy chroma parameters (for chromaVector only)
    size_t m_fftSize = 4096;
    size_t m_hopSize = 2048;
    std::string m_windowType = "hann";
    double m_refFreq = 440.0;

    // Queen Mary progressive key detection state
    std::unique_ptr<GetKeyMode> m_pKeyMode;
    DownmixAndOverlapHelper m_helper;
    size_t m_currentFrame = 0;
    using KeyChange = std::pair<int, double>; // <keyIndex0..23, framePosition>
    std::vector<KeyChange> m_resultKeys;
    int m_prevKey = -1; // -1 invalid

    static int mapQMKeyToIndex24(int qmKey) {
        // QM: 0=no key, 1=C major..12=B major, 13=C minor..24=B minor
        if (qmKey <= 0) return -1;
        if (qmKey <= 24) return qmKey - 1;
        return -1;
    }

    static nlohmann::json makeDefaultFromChroma(const std::vector<double>& chroma) {
        // Fallback picks the max bin as major and assigns low confidence
        static const char* KEY_NAMES[12] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
        size_t idx = 0; double maxv = chroma.empty() ? 0.0 : chroma[0]; double sum = 0.0;
        for (size_t i = 0; i < chroma.size(); ++i) { sum += chroma[i]; if (chroma[i] > maxv) { maxv = chroma[i]; idx = i; } }
        std::string keyName = KEY_NAMES[idx % 12];
        double confidence = (sum > 0.0) ? std::min(1.0, maxv / std::max(1e-12, sum)) : 0.0;
        nlohmann::json chromaJson = nlohmann::json::array();
        for (double v : chroma) chromaJson.push_back(v);
        return nlohmann::json{{"key", keyName}, {"mode", "major"}, {"keyString", keyName}, {"confidence", confidence}, {"chromaVector", chromaJson}};
    }

    static nlohmann::json makeEmptyResult() {
        return nlohmann::json{
            {"key", "C"},
            {"mode", "major"},
            {"keyString", "C"},
            {"confidence", 0.0},
            {"chromaVector", std::vector<double>(12, 0.0)}
        };
    }

    std::vector<double> computeGlobalChroma(const std::vector<float>& mono, float sr) const {
        const size_t N = m_fftSize;
        const size_t H = m_hopSize == 0 ? std::max<size_t>(1, N / 4) : m_hopSize;
        // Window
        std::vector<float> winF;
        if (m_windowType == "hann" || m_windowType.empty()) winF = core::window::hann(N);
        else if (m_windowType == "hamming") winF = core::window::hamming(N);
        else if (m_windowType == "blackman") winF = core::window::blackman(N);
        else winF = core::window::hann(N);
        std::vector<double> window(winF.begin(), winF.end());

        // Number of frames (include last partial, zero-padded)
        size_t numFrames = (mono.size() + H - 1) / H;
        if (numFrames == 0) return std::vector<double>(12, 0.0);

        // FFTW setup
        double* inS = static_cast<double*>(fftw_malloc(sizeof(double) * Nseq));
        fftw_complex* outS = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (Nseq / 2 + 1)));
        if (!in || !out) {
            if (in) fftw_free(in);
            if (out) fftw_free(out);
            return std::vector<double>(12, 0.0);
        }
        fftw_plan plan = fftw_plan_dft_r2c_1d(static_cast<int>(N), in, out, FFTW_ESTIMATE);

        std::vector<double> chroma(12, 0.0);
        const size_t numBins = N / 2 + 1;
        for (size_t f = 0; f < numFrames; ++f) {
            const size_t start = f * H;
            for (size_t i = 0; i < N; ++i) {
                size_t idx = start + i;
                double s = (idx < mono.size()) ? static_cast<double>(mono[idx]) : 0.0;
                in[i] = s * window[i];
            }
            fftw_execute(plan);
            for (size_t k = 1; k < numBins; ++k) {
                double re = out[k][0];
                double im = out[k][1];
                double power = re * re + im * im;
                if (power <= 0.0) continue;
                double fk = (static_cast<double>(k) * sr) / static_cast<double>(N);
                if (fk < 20.0) continue;
                double pitch = 12.0 * std::log2(fk / m_refFreq) + 69.0;
                long nearest = static_cast<long>(std::llround(pitch));
                int chromaBin = static_cast<int>(nearest % 12);
                if (chromaBin < 0) chromaBin += 12;
                chroma[static_cast<size_t>(chromaBin)] += power;
            }
        }
        fftw_destroy_plan(plan);
        fftw_free(in);
        fftw_free(out);
        double sum = 0.0; for (double v : chroma) sum += v;
        if (sum > 0.0) { for (double& v : chroma) v /= sum; }
        return chroma;
    }
};

std::unique_ptr<core::IAnalysisModule> createRealTonalityModule() {
    return std::make_unique<RealTonalityModule>();
}

} } // namespace ave::modules
