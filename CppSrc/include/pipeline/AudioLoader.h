//
// Created by jeanc on 8/27/2025.
//

#ifndef AURASYNC_AUDIOLOADER_H
#define AURASYNC_AUDIOLOADER_H

#include <string>
#include "../core/AudioBuffer.h"

namespace ave::pipeline {

    /**
     * @brief Utility class responsible for loading audio files into an AudioBuffer object.
     *
     * Currently supports loading uncompressed WAV format files.
     */
    class AudioLoader {
    public:
        /**
         * @brief Loads audio data from a specified WAV file path into an AudioBuffer.
         *
         * This method handles common uncompressed WAV formats (PCM 16/24-bit or IEEE float 32-bit),
         * assuming little-endian byte order.
         *
         * @param path The file path to the WAV file.
         * @return A core::AudioBuffer containing the loaded audio data and metadata.
         * @throws std::runtime_error If the file cannot be opened, is corrupted, or if the format is unsupported.
         */
        static core::AudioBuffer loadWav(const std::string& path);
    };

} // namespace ave::pipeline

#endif //AURASYNC_AUDIOLOADER_H