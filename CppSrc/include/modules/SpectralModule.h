//
// Created by jeanc on 8/27/2025.
//

#ifndef AURASYNC_SPECTRALMODULE_H
#define AURASYNC_SPECTRALMODULE_H

#include <memory>

namespace ave { namespace core { class IAnalysisModule; } }

namespace ave { namespace modules {
    /**
     * @brief Factory function to create an instance of the concrete Spectral Analysis module.
     *
     * This module typically handles Short-Time Fourier Transform (STFT),
     * spectral feature extraction (e.g., centroid, rolloff, flux), and band energy calculation.
     * @return A unique pointer to the newly created IAnalysisModule implementation for spectral analysis.
     */
    std::unique_ptr<ave::core::IAnalysisModule> createRealSpectralModule();
} }

#endif //AURASYNC_SPECTRALMODULE_H