#ifndef AVE_MODULES_TONALITYMODULE_H
#define AVE_MODULES_TONALITYMODULE_H

#include <memory>

namespace ave { namespace core { class IAnalysisModule; } }

namespace ave { namespace modules {
    /**
     * @brief Factory function to create an instance of the concrete Tonality (Key Detection) module.
     *
     * This module is responsible for analyzing harmonic content to determine the musical key and related features (e.g., chroma vector).
     * @return A unique pointer to the newly created IAnalysisModule implementation for tonality analysis.
     */
    std::unique_ptr<ave::core::IAnalysisModule> createRealTonalityModule();
} }

#endif // AVE_MODULES_TONALITYMODULE_H