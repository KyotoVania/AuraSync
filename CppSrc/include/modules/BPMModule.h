//
// Created by jeanc on 8/27/2025.
//

#ifndef AURASYNC_BPMMODULE_H
#define AURASYNC_BPMMODULE_H

#include <memory>
#include <nlohmann/json.hpp>

namespace ave {
namespace core { class IAnalysisModule; }
namespace modules {

// Shared BPM configuration (optional)
struct BPMConfig {
    float minBPM = 60.0f;
    float maxBPM = 200.0f;
    size_t frameSize = 1024;   // STFT window size
    size_t hopSize = 512;      // STFT hop size
};

// Factory for the real BPM implementation
std::unique_ptr<core::IAnalysisModule> createRealBPMModule();

} // namespace modules
} // namespace ave

#endif //AURASYNC_BPMMODULE_H