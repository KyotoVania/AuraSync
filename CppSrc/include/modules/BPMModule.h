//
// Created by jeanc on 8/27/2025.
//

#ifndef AURASYNC_BPMMODULE_H
#define AURASYNC_BPMMODULE_H

#include <memory>
#include <nlohmann/json.hpp>

namespace ave {
    namespace core {
        // Forward declaration of the base analysis module interface
        class IAnalysisModule;
    }
    namespace modules {

        /**
         * @brief Configuration structure for the BPM (Beats Per Minute) analysis module.
         */
        struct BPMConfig {
            /** @brief The minimum BPM value to search for (default: 60.0f). */
            float minBPM = 60.0f;

            /** @brief The maximum BPM value to search for (default: 200.0f). */
            float maxBPM = 200.0f;

            /** @brief The window size (in samples) used for Short-Time Fourier Transform (STFT) (default: 1024). */
            size_t frameSize = 1024;

            /** @brief The hop size (in samples) used to advance the STFT window (default: 512). */
            size_t hopSize = 512;
        };

        /**
         * @brief Factory function to create an instance of the concrete BPM analysis module.
         *
         * The concrete implementation of the BPM calculation logic is hidden behind this factory.
         * @return A unique pointer to the newly created IAnalysisModule implementation for BPM analysis.
         */
        std::unique_ptr<core::IAnalysisModule> createRealBPMModule();

    } // namespace modules
} // namespace ave

#endif //AURASYNC_BPMMODULE_H