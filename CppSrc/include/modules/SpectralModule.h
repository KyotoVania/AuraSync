//
// Created by jeanc on 8/27/2025.
//

#ifndef AURASYNC_SPECTRALMODULE_H
#define AURASYNC_SPECTRALMODULE_H

#include <memory>

namespace ave { namespace core { class IAnalysisModule; } }

namespace ave { namespace modules {
    // Factory for the real spectral analysis module
    std::unique_ptr<ave::core::IAnalysisModule> createRealSpectralModule();
} }

#endif //AURASYNC_SPECTRALMODULE_H