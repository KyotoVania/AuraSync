//
// Created by Junie (AI) on 2025-09-16.
//
#ifndef AVE_MODULES_TONALITYMODULE_H
#define AVE_MODULES_TONALITYMODULE_H

#include <memory>

namespace ave { namespace core { class IAnalysisModule; } }

namespace ave { namespace modules {
    // Factory for the real tonality (key detection) module
    std::unique_ptr<ave::core::IAnalysisModule> createRealTonalityModule();
} }

#endif // AVE_MODULES_TONALITYMODULE_H