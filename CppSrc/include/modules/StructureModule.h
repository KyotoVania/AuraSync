//
// Created by Junie (AI) on 2025-09-16.
//
#ifndef AVE_MODULES_STRUCTUREMODULE_H
#define AVE_MODULES_STRUCTUREMODULE_H

#include <memory>

namespace ave { namespace core { class IAnalysisModule; } }

namespace ave { namespace modules {
    // Factory for the real structure (segmentation) module
    std::unique_ptr<ave::core::IAnalysisModule> createRealStructureModule();
} }

#endif // AVE_MODULES_STRUCTUREMODULE_H
