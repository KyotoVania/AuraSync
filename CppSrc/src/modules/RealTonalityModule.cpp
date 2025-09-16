#include "../../include/core/IAnalysisModule.h"
#include "../../include/core/AudioBuffer.h"
#include "../../include/modules/TonalityModule.h"
#include <nlohmann/json.hpp>
#include <fftw3.h>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace ave { namespace modules {

class RealTonalityModule : public core::IAnalysisModule {
public:
    std::string getName() const override { return "Tonality"; }
    std::string getVersion() const override { return "1.0.0"; }
    bool isRealTime() const override { return true; }

    bool initialize(const nlohmann::json& config) override {
        if (config.contains("fftSize")) m_fftSize = std::max<size_t>(256, config["fftSize"].get<size_t>());
        if (config.contains("hopSize")) m_hopSize = std::max<size_t>(1, config["hopSize"].get<size_t>());
        if (m_hopSize > m_fftSize) m_hopSize = m_fftSize / 2; // clamp
        if (config.contains("windowType")) m_windowType = config["windowType"].get<std::string>();
        if (config.contains("referenceFreq")) m_refFreq = std::max(1e-3, config["referenceFreq"].get<double>());
        return true;
    }

    void reset() override {}

    std::vector<std::string> getDependencies() const override { return {}; }

    nlohmann::json process(const core::AudioBuffer& audio, const core::AnalysisContext&) override {
        const float sr = audio.getSampleRate();
        const size_t N = m_fftSize;
        const size_t H = m_hopSize == 0 ? std::max<size_t>(1, N / 4) : m_hopSize;

        // Prepare mono signal
        std::vector<float> mono = audio.getMono();
        if (mono.empty()) {
            return makeEmptyResult();
        }

        // Window
        std::vector<float> winF;
        if (m_windowType == "hann" || m_windowType.empty()) winF = core::window::hann(N);
        else if (m_windowType == "hamming") winF = core::window::hamming(N);
        else if (m_windowType == "blackman") winF = core::window::blackman(N);
        else winF = core::window::hann(N);
        std::vector<double> window(winF.begin(), winF.end());

        // Number of frames (include last partial, zero-padded)
        size_t numFrames = (mono.size() + H - 1) / H;
        if (numFrames == 0) return makeEmptyResult();

        // FFTW setup
        double* in = (double*)fftw_malloc(sizeof(double) * N);
        if (!in) return makeEmptyResult();
        fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
        if (!out) { fftw_free(in); return makeEmptyResult(); }
        fftw_plan plan = fftw_plan_dft_r2c_1d(static_cast<int>(N), in, out, FFTW_ESTIMATE);

        // Accumulate chroma over all frames
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

            for (size_t k = 1; k < numBins; ++k) { // skip DC at k=0
                double re = out[k][0];
                double im = out[k][1];
                double power = re * re + im * im; // power spectrum
                if (power <= 0.0) continue;
                double fk = (static_cast<double>(k) * sr) / static_cast<double>(N);
                if (fk < 20.0) continue; // ignore sub-audio
                // Map to MIDI pitch and chroma bin
                double pitch = 12.0 * std::log2(fk / m_refFreq) + 69.0;
                long nearest = static_cast<long>(std::llround(pitch));
                int chromaBin = static_cast<int>(nearest % 12);
                if (chromaBin < 0) chromaBin += 12;
                chroma[static_cast<size_t>(chromaBin)] += power;
            }
        }

        // Cleanup FFTW
        fftw_destroy_plan(plan);
        fftw_free(in);
        fftw_free(out);

        // Normalize chroma vector (L1 norm)
        double sum = 0.0;
        for (double v : chroma) sum += v;
        if (sum > 0.0) {
            for (double& v : chroma) v = v / sum;
        }

        // Determine key using Krumhansl templates
        static const std::vector<double> profileMajor = {6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.63, 2.24, 2.88};
        static const std::vector<double> profileMinor = {6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17};
        static const char* KEY_NAMES[12] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};

        auto rotate = [](const std::vector<double>& v, int shift){
            std::vector<double> out(v.size());
            const int n = static_cast<int>(v.size());
            for (int i = 0; i < n; ++i) {
                out[static_cast<size_t>((i + shift) % n)] = v[static_cast<size_t>(i)];
            }
            return out;
        };

        auto pearson = [](const std::vector<double>& X, const std::vector<double>& Y){
            const size_t n = X.size();
            double meanX = 0.0, meanY = 0.0;
            for (size_t i = 0; i < n; ++i) { meanX += X[i]; meanY += Y[i]; }
            meanX /= static_cast<double>(n);
            meanY /= static_cast<double>(n);
            double num = 0.0, denX = 0.0, denY = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double dx = X[i] - meanX;
                double dy = Y[i] - meanY;
                num += dx * dy;
                denX += dx * dx;
                denY += dy * dy;
            }
            double den = std::sqrt(denX * denY);
            if (den <= 1e-12) return 0.0;
            return num / den;
        };

        int bestKey = 0; // 0=C, 1=C#, ...
        std::string bestMode = "major";
        double bestScore = -1.0;

        for (int key = 0; key < 12; ++key) {
            auto maj = rotate(profileMajor, key);
            auto min = rotate(profileMinor, key);
            double cMaj = pearson(chroma, maj);
            double cMin = pearson(chroma, min);
            if (cMaj > bestScore) { bestScore = cMaj; bestKey = key; bestMode = "major"; }
            if (cMin > bestScore) { bestScore = cMin; bestKey = key; bestMode = "minor"; }
        }

        std::string keyName = KEY_NAMES[bestKey];
        std::string keyString = (bestMode == "minor") ? (keyName + std::string("m")) : keyName;

        nlohmann::json chromaJson = nlohmann::json::array();
        for (double v : chroma) chromaJson.push_back(v);

        nlohmann::json result = {
            {"key", keyName},
            {"mode", bestMode},
            {"keyString", keyString},
            {"confidence", std::max(0.0, std::min(1.0, bestScore))},
            {"chromaVector", chromaJson}
        };
        return result;
    }

    bool validateOutput(const nlohmann::json& output) const override {
        return output.contains("key") && output.contains("mode") && output.contains("confidence");
    }

private:
    size_t m_fftSize = 4096;
    size_t m_hopSize = 2048;
    std::string m_windowType = "hann";
    double m_refFreq = 440.0;

    static nlohmann::json makeEmptyResult() {
        return nlohmann::json{
            {"key", "C"},
            {"mode", "major"},
            {"keyString", "C"},
            {"confidence", 0.0},
            {"chromaVector", std::vector<double>(12, 0.0)}
        };
    }
};

std::unique_ptr<core::IAnalysisModule> createRealTonalityModule() {
    return std::make_unique<RealTonalityModule>();
}

} } // namespace ave::modules
