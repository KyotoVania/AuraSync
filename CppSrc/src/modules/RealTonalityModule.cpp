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
#include <numeric>

// Queen Mary DSP
#include <dsp/keydetection/GetKeyMode.h>

namespace ave { namespace modules {

/**
 * @brief Lightweight helper inspired by Mixxx DownmixAndOverlapHelper for processing audio frames with overlap.
 *
 * This class handles stereo to mono downmixing and prepares overlapping windows
 * of audio data for further spectral analysis.
 */
class DownmixAndOverlapHelper {
public:
    /// Type definition for the callback function executed when a window is ready.
    using WindowReadyCallback = std::function<bool(double* pBuffer, size_t currentFrame)>;

    /**
     * @brief Initializes the helper with window parameters and a callback function.
     * @param windowSize The size of the analysis window in frames.
     * @param stepSize The step size (hop size) between consecutive windows in frames.
     * @param callback The function to be called when a full window of data is available.
     * @return true if initialization was successful, false otherwise.
     */
    bool initialize(size_t windowSize, size_t stepSize, const WindowReadyCallback& callback) {
        m_buffer.assign(windowSize, 0.0);
        m_windowSize = windowSize;
        m_stepSize = stepSize;
        m_callback = callback;
        // center first frame
        m_writePos = windowSize / 2;
        m_totalFramesProcessed = 0;
        return m_windowSize > 0 && m_stepSize > 0 && m_stepSize <= m_windowSize && static_cast<bool>(m_callback);
    }

    /**
     * @brief Processes input stereo samples, downmixing to mono and filling the internal buffer.
     * @param pInputStereo Pointer to the input stereo buffer (interleaved floats).
     * @param inputStereoSamples The number of total float samples in the input buffer (2 * number of frames).
     * @return true if processing was successful and all callbacks returned true, false otherwise.
     */
    bool processStereoSamples(const float* pInputStereo, size_t inputStereoSamples) {
        const size_t numInputFrames = inputStereoSamples / 2;
        return processInner(pInputStereo, numInputFrames);
    }

    /**
     * @brief Finalizes processing, padding with silence to flush the last partial window.
     * @return true if finalization and last callback execution was successful, false otherwise.
     */
    bool finalize() {
        // Pad with silence to flush last window
        size_size_t framesToFillWindow = (m_windowSize > m_writePos) ? (m_windowSize - m_writePos) : 0;
        size_t numInputFrames = std::max(framesToFillWindow, m_windowSize / 2 - 1);
        return processInner(nullptr, numInputFrames);
    }

private:
    /**
     * @brief Internal processing logic handling input and windowing.
     * @param pInputStereo Pointer to the input stereo buffer, or nullptr for silence padding.
     * @param numInputFrames The number of audio frames to process.
     * @return true if processing was successful, false otherwise.
     */
    bool processInner(const float* pInputStereo, size_t numInputFrames) {
        size_t inRead = 0;
        double* pDownmix = m_buffer.data();

        while (inRead < numInputFrames) {
            size_t readAvailable = numInputFrames - inRead;
            size_t writeAvailable = (m_writePos <= m_windowSize) ? (m_windowSize - m_writePos) : 0;
            size_t numFrames = std::min(readAvailable, writeAvailable);
            if (pInputStereo) {
                for (size_t i = 0; i < numFrames; ++i) {
                    // Downmix stereo to mono (average L/R)
                    pDownmix[m_writePos + i] = (static_cast<double>(pInputStereo[(inRead + i) * 2]) +
                                                static_cast<double>(pInputStereo[(inRead + i) * 2 + 1])) * 0.5;
                }
            } else {
                for (size_t i = 0; i < numFrames; ++i) {
                    // Silence padding
                    pDownmix[m_writePos + i] = 0.0;
                }
            }
            m_writePos += numFrames;
            inRead += numFrames;
            m_totalFramesProcessed += numFrames;

            if (m_writePos == m_windowSize) {
                // Full window ready, execute callback
                bool ok = m_callback(pDownmix, m_totalFramesProcessed);
                if (!ok) return false;
                // Overlap: shift by stepSize
                const size_t overlap = m_windowSize - m_stepSize;
                for (size_t i = 0; i < overlap; ++i) {
                    pDownmix[i] = pDownmix[i + m_stepSize];
                }
                m_writePos -= m_stepSize;
            }
        }
        return true;
    }

    std::vector<double> m_buffer; ///< Internal buffer for the current window.
    size_t m_windowSize = 0; ///< The size of the analysis window in frames.
    size_t m_stepSize = 0; ///< The step size between windows (hop size).
    size_t m_writePos = 0; ///< Current write position in the buffer.
    size_t m_totalFramesProcessed = 0; ///< Total number of frames processed since initialization.
    WindowReadyCallback m_callback; ///< The callback function for a ready window.
};

/**
 * @brief Analysis module for estimating musical key and chromagrams.
 *
 * This module computes a global chroma vector and performs progressive key detection
 * using the Queen Mary DSP library to determine the predominant musical key.
 */
class RealTonalityModule : public core::IAnalysisModule {
public:
    /**
     * @brief Returns the name of the analysis module.
     * @return The module name, "Tonality".
     */
    std::string getName() const override { return "Tonality"; }

    /**
     * @brief Returns the version of the analysis module.
     * @return The module version string.
     */
    std::string getVersion() const override { return "1.1.0"; }

    /**
     * @brief Indicates if the module is designed for real-time processing.
     * @return true, as the module can process audio in chunks.
     */
    bool isRealTime() const override { return true; }

    /**
     * @brief Initializes the module with configuration parameters.
     *
     * Configures FFT parameters (size, hop, window type) and reference frequency.
     * @param config The JSON configuration object.
     * @return true if initialization was successful, false otherwise.
     */
    bool initialize(const nlohmann::json& config) override {
        // Keep legacy chroma config for test compatibility
        if (config.contains("fftSize")) m_fftSize = std::max<size_t>(256, config["fftSize"].get<size_t>());
        if (config.contains("hopSize")) m_hopSize = std::max<size_t>(1, config["hopSize"].get<size_t>());
        // clamp hopSize
        if (m_hopSize > m_fftSize) m_hopSize = m_fftSize / 2;
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

    /**
     * @brief Resets the internal state of the module for a new analysis.
     *
     * Clears all progressive key detection state.
     */
    void reset() override {
        m_prevKey = -1;
        m_resultKeys.clear();
        m_currentFrame = 0;
        m_pKeyMode.reset();
        m_helper = DownmixAndOverlapHelper{};
    }

    /**
     * @brief Returns a list of required dependency modules.
     * @return An empty vector, as this module has no hard dependencies.
     */
    // cppcheck-suppress uselessOverride
    std::vector<std::string> getDependencies() const override { return {}; }

    /**
     * @brief Processes an audio buffer to extract tonality features.
     *
     * Computes a global chroma vector and determines the predominant key
     * and a chroma sequence using Queen Mary's progressive key detection.
     * @param audio The input AudioBuffer to analyze.
     * @param context The analysis context (unused here).
     * @return A JSON object containing the analysis results (key, mode, confidence, chromagrams).
     */
    nlohmann::json process(const core::AudioBuffer& audio, const core::AnalysisContext&) override {
        // Prepare mono signal
        const float sr = audio.getSampleRate();
        std::vector<float> mono = audio.getMono();
        if (mono.empty()) {
            return makeEmptyResult();
        }

        // 1) Compute a simple global chroma vector for tests/compatibility
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
                    // continue, ignore invalid
                    return true;
                }
                if (idx24 != m_prevKey) {
                    m_resultKeys.push_back(std::make_pair(idx24, static_cast<double>(currentFrame)));
                    m_prevKey = idx24;
                }
                return true;
            });
        }

        // Feed audio as fake stereo to helper for downmixing and windowing
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

        // Compute total duration for each key index (0..23)
        std::map<int, double> keyDurations;
        for (size_t i = 0; i < m_resultKeys.size(); ++i) {
            int key = m_resultKeys[i].first;
            double start = m_resultKeys[i].second;
            double end = (i + 1 < m_resultKeys.size()) ? m_resultKeys[i + 1].second : static_cast<double>(m_currentFrame);
            keyDurations[key] += std::max(0.0, end - start);
        }
        int bestKey = -1;
        double maxDur = 0.0;
        // Find the key with the maximum total duration
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
        // Confidence is the fraction of total frames dedicated to the best key
        double confidence = (m_currentFrame > 0) ? (maxDur / static_cast<double>(m_currentFrame)) : 0.0;
        confidence = std::clamp(confidence, 0.0, 1.0);

        nlohmann::json chromaJson = nlohmann::json::array();
        for (double v : chroma) chromaJson.push_back(v);

        // Compute per-frame chromagram sequence aligned with Spectral frames if available
        size_t Nseq = m_fftSize;
        size_t Hseq = m_hopSize == 0 ? std::max<size_t>(1, Nseq / 4) : m_hopSize;
        double frameRate = sr / static_cast<double>(Hseq);
        // Since we don't have Spectral config here, we use local FFT/hop
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

            // Allocate memory for FFTW
            double* inS = static_cast<double*>(fftw_malloc(sizeof(double) * Nseq));
            fftw_complex* outS = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (Nseq / 2 + 1)));

            fftw_plan planS = fftw_plan_dft_r2c_1d(static_cast<int>(Nseq), inS, outS, FFTW_ESTIMATE);
            const size_t numBinsS = Nseq / 2 + 1;

            for (size_t f = 0; f < numFramesSeq; ++f) {
                const size_t start = f * Hseq;
                // Apply windowing and zero-padding
                for (size_t i = 0; i < Nseq; ++i) {
                    size_t idx = start + i;
                    double s = (idx < mono.size()) ? static_cast<double>(mono[idx]) : 0.0;
                    inS[i] = s * window[i];
                }
                fftw_execute(planS);
                std::vector<double> cv(12, 0.0);
                // Compute chromagram from magnitude spectrum
                for (size_t k = 1; k < numBinsS; ++k) {
                    double re = outS[k][0];
                    double im = outS[k][1];
                    double power = re * re + im * im;
                    if (power <= 0.0) continue;
                    double fk = (static_cast<double>(k) * sr) / static_cast<double>(Nseq);
                    if (fk < 20.0) continue;
                    // Pitch calculation based on C=69 (MIDI) and reference frequency
                    double pitch = 12.0 * std::log2(fk / m_refFreq) + 69.0;
                    long nearest = static_cast<long>(std::llround(pitch));
                    // Map MIDI pitch to chroma bin (0=C, 1=C#, ..., 11=B)
                    int cb = static_cast<int>(nearest % 12);
                    if (cb < 0) cb += 12;
                    cv[static_cast<size_t>(cb)] += power;
                }

                // Normalize chroma vector
                double ssum = std::accumulate(cv.begin(), cv.end(), 0.0);
                if (ssum > 0.0) {
                    std::transform(cv.begin(), cv.end(), cv.begin(), [ssum](double v) { return v / ssum; });
                }
                nlohmann::json vj = nlohmann::json::array();
                for (double x : cv) vj.push_back(x);
                // Add frame data to sequence
                chromaSeq.push_back({{"t", static_cast<double>(f) * static_cast<double>(Hseq) / static_cast<double>(sr)}, {"v", vj}});
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

    /**
     * @brief Validates the essential fields of the output JSON object.
     * @param output The JSON object to validate.
     * @return true if the output contains "key", "mode", and "confidence", false otherwise.
     */
    bool validateOutput(const nlohmann::json& output) const override {
        return output.contains("key") && output.contains("mode") && output.contains("confidence");
    }

private:
    // Legacy chroma parameters (used for chromaVector and chromaSequence)
    size_t m_fftSize = 4096; ///< FFT window size in frames.
    size_t m_hopSize = 2048; ///< FFT hop size in frames.
    std::string m_windowType = "hann"; ///< Type of window function to use.
    double m_refFreq = 440.0; ///< Reference frequency for A4 (usually 440 Hz).

    // Queen Mary progressive key detection state
    std::unique_ptr<GetKeyMode> m_pKeyMode; ///< Pointer to the Queen Mary KeyMode detection object.
    DownmixAndOverlapHelper m_helper; ///< Helper for frame downmixing and windowing.
    size_t m_currentFrame = 0; ///< Total frames processed by the module.
    using KeyChange = std::pair<int, double>; ///< Pair of <keyIndex0..23, framePosition>.
    std::vector<KeyChange> m_resultKeys; ///< Sequence of detected key changes.
    int m_prevKey = -1; ///< Index of the key detected in the previous frame (-1 is invalid).

    /**
     * @brief Maps the Queen Mary KeyMode raw key index (1-24) to a 0-23 index.
     *
     * QM key mapping: 0=no key, 1=C major..12=B major, 13=C minor..24=B minor.
     * Target mapping: 0=C major..11=B major, 12=C minor..23=B minor.
     * @param qmKey The raw key index from GetKeyMode.
     * @return The 0-23 key index, or -1 if no key was detected (qmKey=0).
     */
    static int mapQMKeyToIndex24(int qmKey) {
        if (qmKey <= 0) return -1;
        if (qmKey <= 24) return qmKey - 1;
        return -1;
    }

    /**
     * @brief Creates a default result JSON using the global chroma vector as a fallback.
     *
     * Selects the chroma bin with the highest energy as the major key and assigns low confidence.
     * @param chroma The computed global chroma vector.
     * @return A JSON object containing default key/mode and the chroma vector.
     */
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

    /**
     * @brief Creates a JSON object for an empty or failed analysis.
     * @return A JSON object with default key "C major" and 0.0 confidence.
     */
    static nlohmann::json makeEmptyResult() {
        return nlohmann::json{
            {"key", "C"},
            {"mode", "major"},
            {"keyString", "C"},
            {"confidence", 0.0},
            {"chromaVector", std::vector<double>(12, 0.0)}
        };
    }

    /**
     * @brief Computes a single, global chroma vector for the entire mono audio signal.
     *
     * Uses FFTW to compute the magnitude spectrum over all windows, sums the energy
     * into 12 chroma bins, and normalizes the resulting vector.
     * @param mono The input mono audio signal.
     * @param sr The sample rate of the audio.
     * @return A normalized 12-element chroma vector.
     */
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
        double* in = static_cast<double*>(fftw_malloc(sizeof(double) * N));
        fftw_complex* out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1)));

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
            // Sum energy into chroma bins
            for (size_t k = 1; k < numBins; ++k) {
                double re = out[k][0];
                double im = out[k][1];
                double power = re * re + im * im;
                if (power <= 0.0) continue;
                double fk = (static_cast<double>(k) * sr) / static_cast<double>(N);
                if (fk < 20.0) continue;
                // Pitch calculation
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

        // Normalize using accumulate
        double sum = std::accumulate(chroma.begin(), chroma.end(), 0.0);
        if (sum > 0.0) { for (double& v : chroma) v /= sum; }
        return chroma;
    }
};

/**
 * @brief Factory function to create an instance of the RealTonalityModule.
 * @return A unique pointer to the newly created IAnalysisModule instance.
 */
std::unique_ptr<core::IAnalysisModule> createRealTonalityModule() {
    return std::make_unique<RealTonalityModule>();
}

} } // namespace ave::modules