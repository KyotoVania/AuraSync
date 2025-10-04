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

class RealSpectralModule : public core::IAnalysisModule {
public:
    std::string getName() const override { return "Spectral"; }
    std::string getVersion() const override { return "1.0.0"; }
    bool isRealTime() const override { return true; }

    bool initialize(const nlohmann::json& config) override {
        if (config.contains("fftSize")) m_fftSize = std::max<size_t>(32, config["fftSize"].get<size_t>());
        if (config.contains("hopSize")) m_hopSize = std::max<size_t>(1, config["hopSize"].get<size_t>());
        if (m_hopSize > m_fftSize) m_hopSize = m_fftSize / 2; // clamp
        if (config.contains("windowType")) m_windowType = config["windowType"].get<std::string>();
        if (config.contains("bandDefinitions")) {
            m_bandDefs = config["bandDefinitions"];
        } else {
            // Default 5 bands
            m_bandDefs = {
                {"low", {0.0, 250.0}},
                {"lowMid", {250.0, 500.0}},
                {"mid", {500.0, 2000.0}},
                {"highMid", {2000.0, 4000.0}},
                {"high", {4000.0, 22050.0}}
            };
        }
        // Spectral contrast params (tunable)
        if (config.contains("contrastNumBands")) m_contrastNumBands = std::max<int>(1, config["contrastNumBands"].get<int>());
        if (config.contains("contrastMinFreq")) m_contrastMinFreq = std::max(1.0, config["contrastMinFreq"].get<double>());
        if (config.contains("contrastTopPercent")) m_contrastTopPercent = std::clamp(config["contrastTopPercent"].get<double>(), 0.01, 0.9);
        if (config.contains("contrastBottomPercent")) m_contrastBottomPercent = std::clamp(config["contrastBottomPercent"].get<double>(), 0.01, 0.9);

        // Extended spectral timeline mode
        if (config.contains("extendedMode")) m_extendedMode = config["extendedMode"].get<bool>();
        if (config.contains("timelineResolutionHz")) m_timelineResolutionHz = std::max<int>(1, config["timelineResolutionHz"].get<int>());
        return true;
    }

    void reset() override {
        // No persistent FFTW plan kept between process() calls in this simple implementation
    }

    nlohmann::json process(const core::AudioBuffer& audio, const core::AnalysisContext& context) override {
        const float sampleRate = audio.getSampleRate();
        const size_t N = m_fftSize;
        const size_t H = m_hopSize == 0 ? std::max<size_t>(1, N / 4) : m_hopSize;
        if (N < 32) {
            return makeEmptyResult(sampleRate, N, H);
        }

        // Prepare mono signal
        std::vector<float> mono = audio.getMono();
        if (mono.empty()) {
            return makeEmptyResult(sampleRate, N, H);
        }

        // Precompute window
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
        const size_t numFrames = (mono.size() + H - 1) / H; // process until start < mono.size()

        // FFTW setup (plan once, execute per frame)
        double* in = (double*)fftw_malloc(sizeof(double) * N);
        if (!in) return makeEmptyResult(sampleRate, N, H);
        fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
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
        for (size_t k = 0; k < numBins; ++k) {
            double fk = (static_cast<double>(k) * sampleRate) / static_cast<double>(N);
            for (size_t b = 0; b < bandRanges.size(); ++b) {
                if (fk >= bandRanges[b].first && fk <= bandRanges[b].second) {
                    binToBand[k] = static_cast<int>(b);
                    break;
                }
            }
        }

        // Prepare octave-like bands for Spectral Contrast (log-spaced, doubling bandwidth)
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
        for (size_t k = 0; k < numBins; ++k) {
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
        // Mel points
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
        // Build triangular mel filterbank weights W[numMelBands][numBins]
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
        // Precompute DCT-II matrix [numCoeffs x numMelBands]
        std::vector<std::vector<double>> dct(static_cast<size_t>(numCoeffs), std::vector<double>(static_cast<size_t>(numMelBands), 0.0));
        const double scale0 = std::sqrt(1.0 / static_cast<double>(numMelBands));
        const double scale = std::sqrt(2.0 / static_cast<double>(numMelBands));
        for (int c = 0; c < numCoeffs; ++c) {
            for (int m = 0; m < numMelBands; ++m) {
                double val = std::cos(M_PI * static_cast<double>(c) * (static_cast<double>(m) + 0.5) / static_cast<double>(numMelBands));
                dct[static_cast<size_t>(c)][static_cast<size_t>(m)] = (c == 0 ? scale0 : scale) * val;
            }
        }

        // Output structure
        nlohmann::json bands = nlohmann::json::object();
        for (const auto& name : bandNames) bands[name] = nlohmann::json::array();
        nlohmann::json mfcc = nlohmann::json::array();
        // Spectral contrast output: array of {t, v:[...]} with octave-band contrasts
        nlohmann::json spectralContrast = nlohmann::json::array();

        // STFT loop
        for (size_t f = 0; f < numFrames; ++f) {
            const size_t start = f * H;
            // Fill input with windowed samples or zeros beyond length
            for (size_t i = 0; i < N; ++i) {
                size_t idx = start + i;
                double s = (idx < mono.size()) ? static_cast<double>(mono[idx]) : 0.0;
                in[i] = s * window[i];
            }

            // Execute FFT
            fftw_execute(plan);

            // Compute power spectrum vector P
            std::vector<double> P(numBins, 0.0);
            for (size_t k = 0; k < numBins; ++k) {
                double re = out[k][0];
                double im = out[k][1];
                P[k] = re * re + im * im; // power spectrum
            }

            // Aggregate power per band (5-band energies)
            std::vector<double> bandEnergy(bandNames.size(), 0.0);
            for (size_t k = 0; k < numBins; ++k) {
                int b = binToBand[k];
                if (b >= 0) bandEnergy[static_cast<size_t>(b)] += P[k];
            }

            // Compute Mel energies and MFCCs
            std::vector<double> melE(static_cast<size_t>(numMelBands), 0.0);
            for (int m = 0; m < numMelBands; ++m) {
                double s = 0.0;
                for (size_t k = 0; k < numBins; ++k) s += P[k] * melW[static_cast<size_t>(m)][k];
                melE[static_cast<size_t>(m)] = s;
            }
            // log-energy
            for (double& v : melE) v = std::log(std::max(1e-12, v));
            // DCT-II to MFCCs
            std::vector<double> mfccVec(static_cast<size_t>(numCoeffs), 0.0);
            for (int c = 0; c < numCoeffs; ++c) {
                double acc = 0.0;
                for (int m = 0; m < numMelBands; ++m) acc += dct[static_cast<size_t>(c)][static_cast<size_t>(m)] * melE[static_cast<size_t>(m)];
                mfccVec[static_cast<size_t>(c)] = acc;
            }

            // Spectral Contrast per octave band: difference (dB) between top and bottom energy quantiles within band
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
                // Mean of top nTop
                double sumTop = 0.0; for (size_t i = n - nTop; i < n; ++i) sumTop += values[i]; double meanTop = sumTop / static_cast<double>(nTop);
                // Mean of bottom nBot
                double sumBot = 0.0; for (size_t i = 0; i < nBot; ++i) sumBot += values[i]; double meanBot = sumBot / static_cast<double>(nBot);
                double ratio = (meanBot > 0.0 ? (meanTop / meanBot) : (meanTop / eps));
                contrastVec[b] = 10.0 * std::log10(std::max(ratio, 1e-12));
            }

            // Timestamp (frame start in seconds as per spec)
            double t = static_cast<double>(f) * static_cast<double>(H) / static_cast<double>(sampleRate);

            // Append to output arrays
            for (size_t b = 0; b < bandNames.size(); ++b) {
                bands[bandNames[b]].push_back({ {"t", t}, {"v", static_cast<float>(bandEnergy[b])} });
            }
            // Append MFCCs
            nlohmann::json coeffs = nlohmann::json::array();
            for (double c : mfccVec) coeffs.push_back(static_cast<float>(c));
            mfcc.push_back({ {"t", t}, {"v", coeffs} });
            // Append spectral contrast
            nlohmann::json cvec = nlohmann::json::array();
            for (double cv : contrastVec) cvec.push_back(static_cast<float>(cv));
            spectralContrast.push_back({ {"t", t}, {"v", cvec} });
        }

        // Optional extended spectral timeline pass
        nlohmann::json timelineObj;
        if (m_extendedMode) {
            // Define simple 3-band ranges (Hz)
            const double nyq = sampleRate / 2.0;
            const double lowHi = 250.0;
            const double midHi = 4000.0;

            // Precompute simple band mapping per bin
            std::vector<int> binToSimpleBand(numBins, -1); // 0=low,1=mid,2=high
            for (size_t k = 0; k < numBins; ++k) {
                double fk = (static_cast<double>(k) * sampleRate) / static_cast<double>(N);
                if (fk < lowHi) binToSimpleBand[k] = 0;
                else if (fk < midHi) binToSimpleBand[k] = 1;
                else if (fk <= nyq) binToSimpleBand[k] = 2;
            }

            // Timeline hop based on requested resolution
            const double desiredRes = static_cast<double>(std::max(1, m_timelineResolutionHz));
            const size_t hopT = std::max<size_t>(1, static_cast<size_t>(std::floor(static_cast<double>(sampleRate) / desiredRes)));
            const size_t numFramesT = (mono.size() + hopT - 1) / hopT;

            nlohmann::json timelineBands = nlohmann::json::object();
            timelineBands["low"] = nlohmann::json::array();
            timelineBands["mid"] = nlohmann::json::array();
            timelineBands["high"] = nlohmann::json::array();

            for (size_t f = 0; f < numFramesT; ++f) {
                const size_t start = f * hopT;
                // Fill FFT input for this timeline frame
                for (size_t i = 0; i < N; ++i) {
                    size_t idx = start + i;
                    double s = (idx < mono.size()) ? static_cast<double>(mono[idx]) : 0.0;
                    in[i] = s * window[i];
                }
                fftw_execute(plan);

                // Power spectrum
                std::vector<double> P(numBins, 0.0);
                for (size_t k = 0; k < numBins; ++k) {
                    double re = out[k][0];
                    double im = out[k][1];
                    P[k] = re * re + im * im;
                }
                // Aggregate low/mid/high
                double eLow = 0.0, eMid = 0.0, eHigh = 0.0;
                for (size_t k = 0; k < numBins; ++k) {
                    int b = binToSimpleBand[k];
                    if (b == 0) eLow += P[k];
                    else if (b == 1) eMid += P[k];
                    else if (b == 2) eHigh += P[k];
                }
                timelineBands["low"].push_back(static_cast<float>(eLow));
                timelineBands["mid"].push_back(static_cast<float>(eMid));
                timelineBands["high"].push_back(static_cast<float>(eHigh));
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
        // Pack contrast band edges for reference/consumers
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

    bool validateOutput(const nlohmann::json& output) const override {
        return output.contains("bands") && output.contains("frameRate");
    }

private:
    size_t m_fftSize = 2048;
    size_t m_hopSize = 512;
    std::string m_windowType = "hann";
    nlohmann::json m_bandDefs = nlohmann::json::object();

    // Extended spectral timeline parameters
    bool m_extendedMode = false;
    int m_timelineResolutionHz = 100; // points per second

    // Spectral contrast parameters (tunable)
    int m_contrastNumBands = 6;       // number of octave-like bands starting at m_contrastMinFreq
    double m_contrastMinFreq = 60.0;  // Hz, start of first band
    double m_contrastTopPercent = 0.2;    // fraction of top energies used as peaks
    double m_contrastBottomPercent = 0.2; // fraction of bottom energies used as valleys

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

std::unique_ptr<core::IAnalysisModule> createRealSpectralModule() {
    return std::make_unique<RealSpectralModule>();
}

} // namespace modules
} // namespace ave
