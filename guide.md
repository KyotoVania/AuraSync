# Guide: Porting BPM Algorithm from Rust to C++

## 📊 Pôle 2: Remplacement du Fake BPM par l'équivalent Rust

### Architecture de l'algorithme Rust existant

Le code Rust utilise une approche en 4 étapes:

1. **ODF (Onset Detection Function)** - Génération via flux spectral
2. **ACF (Autocorrelation Function)** - Via théorème de Wiener-Khinchin
3. **Peak Detection** - Recherche de pics dans la plage BPM
4. **Stabilization** - Médiane mobile sur historique

### Mapping Rust → C++

| Composant Rust | Équivalent C++ | Bibliothèque |
|---------------|----------------|--------------|
| `rustfft` | FFTW3 | Déjà intégré |
| `num_complex::Complex` | `fftw_complex` | FFTW3 |
| `Vec<f32>` | `std::vector<float>` | STL |
| `hound` (WAV reader) | libsndfile | À ajouter |
| `dasp_window` | Implémentation manuelle | Dans AudioBuffer.h |

### Structure du module BPM réel

```cpp
// CppAnalysis/src/modules/RealBPMModule.cpp

#include "../../include/modules/BPMModule.h"
#include <fftw3.h>
#include <vector>
#include <algorithm>
#include <cmath>

namespace ave::modules {

class RealBPMModule : public core::IAnalysisModule {
private:
    // Paramètres
    float m_minBPM = 60.0f;
    float m_maxBPM = 200.0f;
    
    // ODF parameters (from Rust)
    size_t m_frameSize = 2048;
    size_t m_hopSize = 512;
    
    // ACF parameters
    size_t m_acfWindowSize;
    
    // Stabilization
    std::vector<float> m_bpmHistory;
    size_t m_historySize = 10;
    
    // FFTW plans (optimisation)
    fftw_plan m_fftPlan;
    fftw_plan m_ifftPlan;
    fftw_complex* m_fftBuffer;
    
public:
    // Implementation details below...
};

}
```

### Étapes de portage

#### 1️⃣ **ODF Generation** (odf.rs → C++)

```cpp
std::vector<float> generateODF(const float* samples, size_t numSamples) {
    std::vector<float> odf;
    std::vector<float> window = core::window::hann(m_frameSize);
    
    // Previous magnitudes for spectral flux
    std::vector<float> prevMagnitudes(m_frameSize / 2 + 1, 0.0f);
    
    // Allocate FFTW buffers
    double* input = (double*)fftw_malloc(sizeof(double) * m_frameSize);
    fftw_complex* output = (fftw_complex*)fftw_malloc(
        sizeof(fftw_complex) * (m_frameSize / 2 + 1));
    
    fftw_plan plan = fftw_plan_dft_r2c_1d(
        m_frameSize, input, output, FFTW_ESTIMATE);
    
    size_t numFrames = (numSamples - m_frameSize) / m_hopSize + 1;
    
    for (size_t frame = 0; frame < numFrames; ++frame) {
        size_t start = frame * m_hopSize;
        
        // Apply window and copy to FFTW buffer
        for (size_t i = 0; i < m_frameSize; ++i) {
            input[i] = samples[start + i] * window[i];
        }
        
        // FFT
        fftw_execute(plan);
        
        // Calculate magnitudes
        std::vector<float> magnitudes;
        for (size_t i = 0; i <= m_frameSize / 2; ++i) {
            float real = output[i][0];
            float imag = output[i][1];
            magnitudes.push_back(std::sqrt(real * real + imag * imag));
        }
        
        // Spectral flux with log scale (comme dans Rust)
        std::vector<float> spectralDiffs;
        for (size_t i = 0; i < magnitudes.size(); ++i) {
            float logCurrent = std::log(magnitudes[i] + 1e-10f);
            float logPrev = std::log(prevMagnitudes[i] + 1e-10f);
            float diff = std::max(0.0f, logCurrent - logPrev);
            spectralDiffs.push_back(diff);
        }
        
        // Median aggregation
        std::sort(spectralDiffs.begin(), spectralDiffs.end());
        float median = spectralDiffs[spectralDiffs.size() / 2];
        odf.push_back(median);
        
        prevMagnitudes = magnitudes;
    }
    
    fftw_destroy_plan(plan);
    fftw_free(input);
    fftw_free(output);
    
    return odf;
}
```

#### 2️⃣ **ACF Calculation** (acf.rs → C++)

```cpp
std::vector<float> calculateACF(const std::vector<float>& odf) {
    size_t windowSize = std::min(m_acfWindowSize, odf.size());
    size_t fftLen = nextPowerOfTwo(windowSize * 2);
    
    // Prepare complex buffer with zero padding
    fftw_complex* buffer = (fftw_complex*)fftw_malloc(
        sizeof(fftw_complex) * fftLen);
    
    // Copy ODF to complex buffer
    for (size_t i = 0; i < windowSize; ++i) {
        buffer[i][0] = odf[odf.size() - windowSize + i]; // Real
        buffer[i][1] = 0.0; // Imaginary
    }
    // Zero padding
    for (size_t i = windowSize; i < fftLen; ++i) {
        buffer[i][0] = 0.0;
        buffer[i][1] = 0.0;
    }
    
    // Forward FFT
    fftw_plan fft = fftw_plan_dft_1d(
        fftLen, buffer, buffer, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(fft);
    
    // Power spectrum |X(f)|²
    for (size_t i = 0; i < fftLen; ++i) {
        float power = buffer[i][0] * buffer[i][0] + 
                     buffer[i][1] * buffer[i][1];
        buffer[i][0] = power;
        buffer[i][1] = 0.0;
    }
    
    // Inverse FFT
    fftw_plan ifft = fftw_plan_dft_1d(
        fftLen, buffer, buffer, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(ifft);
    
    // Extract ACF and normalize
    std::vector<float> acf(windowSize);
    float scale = 1.0f / fftLen;
    for (size_t i = 0; i < windowSize; ++i) {
        acf[i] = buffer[i][0] * scale;
    }
    
    fftw_destroy_plan(fft);
    fftw_destroy_plan(ifft);
    fftw_free(buffer);
    
    return acf;
}
```

#### 3️⃣ **BPM Estimation** (estimator.rs → C++)

```cpp
float estimateBPM(const std::vector<float>& acf, float odfSampleRate) {
    // Convert BPM range to lag range
    float maxPeriod = 60.0f / m_minBPM;
    float minPeriod = 60.0f / m_maxBPM;
    
    size_t minLag = static_cast<size_t>(minPeriod * odfSampleRate);
    size_t maxLag = static_cast<size_t>(maxPeriod * odfSampleRate);
    
    minLag = std::max(size_t(1), minLag);
    maxLag = std::min(acf.size() - 1, maxLag);
    
    // Find peak in lag range
    float maxValue = 0.0f;
    size_t peakLag = minLag;
    
    for (size_t lag = minLag; lag <= maxLag; ++lag) {
        if (acf[lag] > maxValue) {
            maxValue = acf[lag];
            peakLag = lag;
        }
    }
    
    // Quality check
    if (maxValue < 0.1f) {
        return 0.0f; // No valid BPM found
    }
    
    // Convert lag to BPM
    float bpm = (odfSampleRate * 60.0f) / peakLag;
    
    // Stabilization with median
    m_bpmHistory.push_back(bpm);
    if (m_bpmHistory.size() > m_historySize) {
        m_bpmHistory.erase(m_bpmHistory.begin());
    }
    
    // Calculate median
    std::vector<float> sorted = m_bpmHistory;
    std::sort(sorted.begin(), sorted.end());
    float medianBPM = sorted[sorted.size() / 2];
    
    return medianBPM;
}
```

### Optimisations C++

1. **Plan FFTW réutilisable**: Créer les plans une fois dans `initialize()`
2. **Buffer pooling**: Réutiliser les buffers alloués
3. **SIMD**: FFTW utilise automatiquement SSE/AVX
4. **Multithreading**: `fftw_plan_with_nthreads()` pour paralléliser

### Tests de validation

Comparer les résultats avec l'implémentation Rust:

```cpp
// tests/test_bpm_accuracy.cpp
TEST_CASE("BPM matches Rust implementation") {
    // Load same test file
    AudioBuffer testAudio = loadTestFile("test_130bpm.wav");
    
    // Run C++ version
    RealBPMModule cppModule;
    auto cppResult = cppModule.process(testAudio, context);
    
    // Compare with Rust output (from JSON)
    auto rustResult = loadRustResult("test_130bpm_rust.json");
    
    REQUIRE(std::abs(cppResult["bpm"] - rustResult["bpm"]) < 0.5f);
}
```

### Intégration progressive

1. **Phase 1**: Porter uniquement l'ODF et valider avec des tests
2. **Phase 2**: Ajouter ACF et validation
3. **Phase 3**: BPM estimation complète
4. **Phase 4**: Optimisations et benchmarks

### Commandes de build

```bash
# Build avec module réel BPM
mkdir build && cd build
cmake .. -DUSE_REAL_BPM=ON
make -j8

# Test du module
./bin/ave_analysis test_audio.wav output.json

# Benchmark Fake vs Real
./bin/benchmark_bpm
```

### Résultat attendu

Le module réel devrait:
- Détecter le BPM avec ±2% de précision
- Processing < 100ms pour 3min d'audio
- Générer la même structure JSON que le fake
- Supporter 60-200 BPM (extensible)