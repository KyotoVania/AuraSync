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

/**
 * @brief Real Spectral Analysis Module.
 *
 * This module performs Short-Time Fourier Transform (STFT) on the audio signal
 * to extract various spectral features over time, including:
 * 1. Band energies based on configurable frequency bands.
 * 2. Mel-Frequency Cepstral Coefficients (MFCCs).
 * 3. Spectral Contrast.
 * 4. An optional high-resolution spectral timeline.
 */
class RealSpectralModule : public core::IAnalysisModule {
public:
    /**
     * @brief Get the module's name.
     * @return The string "Spectral".
     */
    std::string getName() const override { return "Spectral"; }

    /**
     * @brief Get the module's version.
     * @return The string "1.0.2".
     */
    std::string getVersion() const override { return "1.0.2"; }

    /**
     * @brief Indicate if the module supports real-time processing.
     * @return \c true, as this module can process in real-time.
     */
    bool isRealTime() const override { return true; }

    /**
     * @brief Initialize the module with configuration parameters.
     * @param config A JSON object containing configuration values.
     * @return \c true if initialization was successful, \c false otherwise.
     */
    bool initialize(const nlohmann::json& config) override {
        // FFT Parameters
        if (config.contains("fftSize")) m_fftSize = std::max<size_t>(32, config["fftSize"].get<size_t>());
        if (config.contains("hopSize")) m_hopSize = std::max<size_t>(1, config["hopSize"].get<size_t>());
        if (m_hopSize > m_fftSize) m_hopSize = m_fftSize / 2; // clamp
        if (config.contains("windowType")) m_windowType = config["windowType"].get<std::string>();

        // Band Definitions
        if (config.contains("bandDefinitions")) {
            m_bandDefs = config["bandDefinitions"];
        } else {
            // Default 3 bands for backward compatibility
            m_bandDefs = {
                {"low", {0.0, 250.0}},
                {"mid", {250.0, 4000.0}},
                {"high", {4000.0, 20000.0}}
            };
        }

        // Spectral Contrast Parameters
        if (config.contains("contrastNumBands")) m_contrastNumBands = std::max<int>(1, config["contrastNumBands"].get<int>());
        if (config.contains("contrastMinFreq")) m_contrastMinFreq = std::max(1.0, config["contrastMinFreq"].get<double>());
        if (config.contains("contrastTopPercent")) m_contrastTopPercent = std::clamp(config["contrastTopPercent"].get<double>(), 0.01, 0.9);
        if (config.contains("contrastBottomPercent")) m_contrastBottomPercent = std::clamp(config["contrastBottomPercent"].get<double>(), 0.01, 0.9);

        // Extended Spectral Timeline Mode
        if (config.contains("extendedMode")) m_extendedMode = config["extendedMode"].get<bool>();
        if (config.contains("timelineResolutionHz")) m_timelineResolutionHz = std::max<int>(1, config["timelineResolutionHz"].get<int>());

        return true;
    }

    /**
     * @brief Reset the module's internal state.
     */
    void reset() override {
        // No persistent FFTW plan kept between process() calls in this simple implementation
    }

    /**
     * @brief Processes the audio buffer to compute spectral features frame by frame.
     *
     * @param audio The input audio buffer.
     * @param context The analysis context (used for sample rate).
     * @return A JSON object containing spectral bands, MFCCs, spectral contrast, and metadata.
     */
    nlohmann::json process(const core::AudioBuffer& audio, const core::AnalysisContext& context) override {
        const float sampleRate = audio.getSampleRate();
        const size_t N = m_fftSize;
        const size_t H = m_hopSize == 0 ? std::max<size_t>(1, N / 4) : m_hopSize;

        if (N < 32) return makeEmptyResult(sampleRate, N, H);

        // Prepare mono signal
        std::vector<float> mono = audio.getMono();
        if (mono.empty()) return makeEmptyResult(sampleRate, N, H);

        // Precompute window function
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
        const size_t numFrames = (mono.size() + H - 1) / H;

        // FFTW setup (plan once, execute per frame)
        double* in = static_cast<double*>(fftw_malloc(sizeof(double) * N));
        if (!in) return makeEmptyResult(sampleRate, N, H);

        fftw_complex* out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1)));
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

        // Extract band definitions and ensure valid ranges
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
        std::vector<int> bandBinCounts(bandNames.size(), 0);

        // Map bins to bands
        // Start at k=1 to exclude DC (0Hz) from band mapping
        for (size_t k = 1; k < numBins; ++k) {
            double fk = (static_cast<double>(k) * sampleRate) / static_cast<double>(N);
            for (size_t b = 0; b < bandRanges.size(); ++b) {
                if (fk >= bandRanges[b].first && fk <= bandRanges[b].second) {
                    binToBand[k] = static_cast<int>(b);
                    bandBinCounts[b]++;
                    break;
                }
            }
        }

        // Prepare octave-like bands for Spectral Contrast (log-spaced)
        std::vector<std::pair<double,double>> contrastBandEdges;
        {
            double flo = std::max(1.0, m_contrastMinFreq);
            for (int i = 0; i < m_contrastNumBands && flo < nyquist; ++i) {
                double fhi = std::min(nyquist, flo * 2.0);
                if (fhi <= flo) break;
                contrastBandEdges.emplace_back(flo, fhi);
                flo = fhi;
            }
            if (contrastBandEdges.empty()) {
                contrastBandEdges.emplace_back(60.0, nyquist);
            }
        }

        // Map FFT bins to contrast bands
        std::vector<std::vector<size_t>> contrastBandBins(contrastBandEdges.size());
        for (size_t k = 1; k < numBins; ++k) { // Also skip DC here
            double fk = (static_cast<double>(k) * sampleRate) / static_cast<double>(N);
            for (size_t b = 0; b < contrastBandEdges.size(); ++b) {
                const auto& be = contrastBandEdges[b];
                bool inLast = (b + 1 == contrastBandEdges.size());
                if ((fk >= be.first && fk < be.second) || (inLast && fk <= be.second)) {
                    contrastBandBins[b].push_back(k);
                    break;
                }
            }
        }

        // Prepare MFCC computation (Mel filterbank + DCT-II)
        const int numMelBands = 40;
        const int numCoeffs = 13;
        const double fmin = 20.0;
        const double fmax = nyquist;

        auto hzToMel = [](double f) { return 2595.0 * std::log10(1.0 + f / 700.0); };
        auto melToHz = [](double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); };

        std::vector<double> melPoints(numMelBands + 2);
        double melMin = hzToMel(fmin);
        double melMax = hzToMel(fmax);

        for (int i = 0; i < numMelBands + 2; ++i) {
            melPoints[static_cast<size_t>(i)] = melMin + (melMax - melMin) * (static_cast<double>(i) / static_cast<double>(numMelBands + 1));
        }

        // Convert mel points to FFT bin indices
        std::vector<size_t> binPoints(numMelBands + 2);
        for (int i = 0; i < numMelBands + 2; ++i) {
            double hz = melToHz(melPoints[static_cast<size_t>(i)]);
            size_t bin = static_cast<size_t>(std::floor((hz * static_cast<double>(N)) / static_cast<double>(sampleRate)));
            if (bin >= numBins) bin = numBins - 1;
            binPoints[static_cast<size_t>(i)] = bin;
        }

        // Build triangular mel filterbank weights
        std::vector<std::vector<double>> melW(static_cast<size_t>(numMelBands), std::vector<double>(numBins, 0.0));
        for (int m = 0; m < numMelBands; ++m) {
            size_t a = binPoints[static_cast<size_t>(m)];
            size_t b = binPoints[static_cast<size_t>(m + 1)];
            size_t c = binPoints[static_cast<size_t>(m + 2)];
            if (a == b) { if (b > 0) --a; }
            if (b == c) { if (c + 1 < numBins) ++c; }

            for (size_t k = a; k < b; ++k) {
                double w = (b == a) ? 0.0 : (static_cast<double>(k) - static_cast<double>(a)) / (static_cast<double>(b) - static_cast<double>(a));
                melW[static_cast<size_t>(m)][k] = std::max(0.0, std::min(1.0, w));
            }
            for (size_t k = b; k <= c && k < numBins; ++k) {
                double w = (c == b) ? 0.0 : (static_cast<double>(c) - static_cast<double>(k)) / (static_cast<double>(c) - static_cast<double>(b));
                melW[static_cast<size_t>(m)][k] = std::max(0.0, std::min(1.0, w));
            }
        }

        // Precompute DCT-II matrix
        std::vector<std::vector<double>> dct(static_cast<size_t>(numCoeffs), std::vector<double>(static_cast<size_t>(numMelBands), 0.0));
        const double scale0 = std::sqrt(1.0 / static_cast<double>(numMelBands));
        const double scale = std::sqrt(2.0 / static_cast<double>(numMelBands));
        for (int c = 0; c < numCoeffs; ++c) {
            for (int m = 0; m < numMelBands; ++m) {
                double val = std::cos(M_PI * static_cast<double>(c) * (static_cast<double>(m) + 0.5) / static_cast<double>(numMelBands));
                dct[static_cast<size_t>(c)][static_cast<size_t>(m)] = (c == 0 ? scale0 : scale) * val;
            }
        }

        // Output containers
        nlohmann::json bands = nlohmann::json::object();
        for (const auto& name : bandNames) bands[name] = nlohmann::json::array();
        nlohmann::json mfcc = nlohmann::json::array();
        nlohmann::json spectralContrast = nlohmann::json::array();

        // ---------------- STFT Processing Loop ----------------
        for (size_t f = 0; f < numFrames; ++f) {
            const size_t start = f * H;
            // Fill input with windowed samples
            for (size_t i = 0; i < N; ++i) {
                size_t idx = start + i;
                double s = (idx < mono.size()) ? static_cast<double>(mono[idx]) : 0.0;
                in[i] = s * window[i];
            }

            // Execute FFT
            fftw_execute(plan);

            // Compute power spectrum
            std::vector<double> P(numBins, 0.0);
            for (size_t k = 0; k < numBins; ++k) {
                double re = out[k][0];
                double im = out[k][1];
                P[k] = re * re + im * im;
            }

            // --- 1. Band Energies ---
            std::vector<double> bandEnergy(bandNames.size(), 0.0);
            // Loop from k=1 to exclude DC from energy sum
            for (size_t k = 1; k < numBins; ++k) {
                int b = binToBand[k];
                if (b >= 0) bandEnergy[static_cast<size_t>(b)] += P[k];
            }
            // Normalize by bin count (density)
            for (size_t b = 0; b < bandNames.size(); ++b) {
                if (bandBinCounts[b] > 0) {
                    bandEnergy[b] /= static_cast<double>(bandBinCounts[b]);
                }
            }

            // --- 2. MFCCs ---
            std::vector<double> melE(static_cast<size_t>(numMelBands), 0.0);
            for (int m = 0; m < numMelBands; ++m) {
                double s = 0.0;
                for (size_t k = 0; k < numBins; ++k) s += P[k] * melW[static_cast<size_t>(m)][k];
                melE[static_cast<size_t>(m)] = std::log(std::max(1e-12, s));
            }
            std::vector<double> mfccVec(static_cast<size_t>(numCoeffs), 0.0);
            for (int c = 0; c < numCoeffs; ++c) {
                double acc = 0.0;
                for (int m = 0; m < numMelBands; ++m) acc += dct[static_cast<size_t>(c)][static_cast<size_t>(m)] * melE[static_cast<size_t>(m)];
                mfccVec[static_cast<size_t>(c)] = acc;
            }

            // --- 3. Spectral Contrast ---
            std::vector<double> contrastVec(contrastBandBins.size(), 0.0);
            const double eps = 1e-12;
            for (size_t b = 0; b < contrastBandBins.size(); ++b) {
                const auto& bins = contrastBandBins[b];
                if (bins.empty()) { contrastVec[b] = 0.0; continue; }
                std::vector<double> values; values.reserve(bins.size());
                for (size_t k : bins) values.push_back(P[k] + eps);
                std::sort(values.begin(), values.end());
                size_t n = values.size();
                size_t nTop = static_cast<size_t>(std::max(1.0, std::floor(m_contrastTopPercent * static_cast<double>(n))));
                size_t nBot = static_cast<size_t>(std::max(1.0, std::floor(m_contrastBottomPercent * static_cast<double>(n))));

                double sumTop = 0.0;
                for (size_t i = n - nTop; i < n; ++i) sumTop += values[i];
                double meanTop = sumTop / static_cast<double>(nTop);

                double sumBot = 0.0;
                for (size_t i = 0; i < nBot; ++i) sumBot += values[i];
                double meanBot = sumBot / static_cast<double>(nBot);

                double ratio = (meanBot > 0.0 ? (meanTop / meanBot) : (meanTop / eps));
                contrastVec[b] = 10.0 * std::log10(std::max(ratio, eps));
            }

            // Append to output
            double t = static_cast<double>(f) * static_cast<double>(H) / static_cast<double>(sampleRate);

            for (size_t b = 0; b < bandNames.size(); ++b) {
                bands[bandNames[b]].push_back({ {"t", t}, {"v", static_cast<float>(bandEnergy[b])} });
            }

            nlohmann::json coeffs = nlohmann::json::array();
            for (double c : mfccVec) coeffs.push_back(static_cast<float>(c));
            mfcc.push_back({ {"t", t}, {"v", coeffs} });

            nlohmann::json cvec = nlohmann::json::array();
            for (double cv : contrastVec) cvec.push_back(static_cast<float>(cv));
            spectralContrast.push_back({ {"t", t}, {"v", cvec} });
        }

        // ---------------- Optional Extended Timeline ----------------
        nlohmann::json timelineObj;
        if (m_extendedMode) {
            const double desiredRes = static_cast<double>(std::max(1, m_timelineResolutionHz));
            const size_t hopT = std::max<size_t>(1, static_cast<size_t>(std::floor(static_cast<double>(sampleRate) / desiredRes)));
            const size_t numFramesT = (mono.size() + hopT - 1) / hopT;

            nlohmann::json timelineBands = nlohmann::json::object();
            for (const auto& name : bandNames) {
                timelineBands[name] = nlohmann::json::array();
            }

            for (size_t f = 0; f < numFramesT; ++f) {
                const size_t start = f * hopT;
                for (size_t i = 0; i < N; ++i) {
                    size_t idx = start + i;
                    double s = (idx < mono.size()) ? static_cast<double>(mono[idx]) : 0.0;
                    in[i] = s * window[i];
                }
                fftw_execute(plan);

                std::vector<double> P(numBins, 0.0);
                for (size_t k = 0; k < numBins; ++k) {
                    P[k] = out[k][0]*out[k][0] + out[k][1]*out[k][1];
                }

                std::vector<double> e(bandNames.size(), 0.0);
                // Ignore DC bin here too
                for (size_t k = 1; k < numBins; ++k) {
                    int b = binToBand[k];
                    if (b >= 0) e[static_cast<size_t>(b)] += P[k];
                }
                for (size_t b = 0; b < bandNames.size(); ++b) {
                    if (bandBinCounts[b] > 0) e[b] /= static_cast<double>(bandBinCounts[b]);
                    timelineBands[bandNames[b]].push_back(static_cast<float>(e[b]));
                }
            }

            timelineObj = {
                {"resolutionHz", std::max(1, m_timelineResolutionHz)},
                {"bands", timelineBands}
            };
        }

        // Cleanup
        fftw_destroy_plan(plan);
        fftw_free(in);
        fftw_free(out);

        const double frameRate = static_cast<double>(sampleRate) / static_cast<double>(H);

        nlohmann::json contrastBandsMeta = nlohmann::json::array();
        for (const auto& be : contrastBandEdges) contrastBandsMeta.push_back({be.first, be.second});

        nlohmann::json result = {
            {"bands", bands},
            {"mfcc", mfcc},
            {"mfccOrder", 13},
            {"spectralContrast", spectralContrast},
            {"spectralContrastBands", contrastBandsMeta},
            {"fftSize", N},
            {"hopSize", H},
            {"frameRate", frameRate}
        };

        if (m_extendedMode) {
            result["spectralTimeline"] = timelineObj;
        }
        return result;
    }

    /**
     * @brief Validates the structure of the module's output JSON.
     * @param output The JSON object produced by the process function.
     * @return \c true if the output contains the mandatory fields, \c false otherwise.
     */
    bool validateOutput(const nlohmann::json& output) const override {
        return output.contains("bands") && output.contains("frameRate");
    }

private:
    size_t m_fftSize = 2048; ///< @brief Size of the FFT window in samples.
    size_t m_hopSize = 512;  ///< @brief Hop size between consecutive FFT windows in samples.
    std::string m_windowType = "hann"; ///< @brief Type of window function (e.g., "hann", "hamming").
    nlohmann::json m_bandDefs = nlohmann::json::object(); ///< @brief Definitions of custom frequency bands for energy calculation.

    // Extended spectral timeline parameters
    bool m_extendedMode = false; ///< @brief Flag to enable or disable the high-resolution spectral timeline.
    int m_timelineResolutionHz = 100; ///< @brief Desired temporal resolution for the timeline in frames per second.

    // Spectral contrast parameters
    int m_contrastNumBands = 6; ///< @brief Number of bands for the spectral contrast calculation.
    double m_contrastMinFreq = 60.0; ///< @brief Minimum frequency for the spectral contrast bands.
    double m_contrastTopPercent = 0.2; ///< @brief Top percentage of energy used to compute the peak (numerator) for contrast.
    double m_contrastBottomPercent = 0.2; ///< @brief Bottom percentage of energy used to compute the valley (denominator) for contrast.

    /**
     * @brief Creates a minimal JSON result object for error or empty input cases.
     * @param sampleRate The sample rate of the audio.
     * @param N The FFT size.
     * @param H The hop size.
     * @return A basic JSON object with bands set to empty object and metadata.
     */
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

/**
 * @brief Factory function to create an instance of the RealSpectralModule.
 * @return A unique pointer to the newly created module.
 */
std::unique_ptr<core::IAnalysisModule> createRealSpectralModule() {
    return std::make_unique<RealSpectralModule>();
}

} // namespace modules
} // namespace ave