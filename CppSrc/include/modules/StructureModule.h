//
// Created by Junie (AI) on 2025-09-16.
//
#ifndef AVE_MODULES_STRUCTUREMODULE_H
#define AVE_MODULES_STRUCTUREMODULE_H

#include <memory>

namespace ave { namespace core { class IAnalysisModule; } }

namespace ave { namespace modules {
    /**
     * @brief Factory function to create an instance of the concrete Structure (Segmentation) module.
     *
     * This module is responsible for detecting and labeling musical sections or segments within the audio.
     * @return A unique pointer to the newly created IAnalysisModule implementation for structure analysis.
     */
    std::unique_ptr<ave::core::IAnalysisModule> createRealStructureModule();
} }

#endif // AVE_MODULES_STRUCTUREMODULE_H