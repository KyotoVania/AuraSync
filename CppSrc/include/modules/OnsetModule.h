//
// Created by Junie (AI) on 2025-09-15.
//
#ifndef AURASYNC_ONSETMODULE_H
#define AURASYNC_ONSETMODULE_H

#include <memory>

namespace ave { namespace core { class IAnalysisModule; } }

namespace ave { namespace modules {
    // Factory for the real onset detection module
    std::unique_ptr<ave::core::IAnalysisModule> createRealOnsetModule();
} }

#endif // AURASYNC_ONSETMODULE_H
