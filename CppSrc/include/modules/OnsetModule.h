#ifndef AVE_MODULES_ONSETMODULE_H
#define AVE_MODULES_ONSETMODULE_H

#include <memory>

namespace ave { namespace core { class IAnalysisModule; } }

namespace ave { namespace modules {
    /**
     * @brief Factory function to create an instance of the concrete Onset Detection analysis module.
     *
     * This function provides the implementation for detecting musical onsets (note starts).
     * @return A unique pointer to the newly created IAnalysisModule implementation for onset detection.
     */
    std::unique_ptr<ave::core::IAnalysisModule> createRealOnsetModule();
} }

#endif // AVE_MODULES_ONSETMODULE_H