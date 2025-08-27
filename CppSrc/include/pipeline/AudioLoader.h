//
// Created by jeanc on 8/27/2025.
//

#ifndef AURASYNC_AUDIOLOADER_H
#define AURASYNC_AUDIOLOADER_H

#include <string>
#include "../core/AudioBuffer.h"

namespace ave::pipeline {

class AudioLoader {
public:
    // Load a WAV file (PCM 16/24-bit or IEEE float 32-bit), little-endian.
    // Throws std::runtime_error on failure.
    static core::AudioBuffer loadWav(const std::string& path);
};

} // namespace ave::pipeline

#endif //AURASYNC_AUDIOLOADER_H