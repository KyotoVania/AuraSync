#include "../../include/pipeline/AudioLoader.h"
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring> // Required for std::memcpy

namespace ave::pipeline {

/**
 * @brief Reads a 32-bit unsigned integer from the file stream in little-endian format.
 * @param f The input file stream.
 * @return The 32-bit unsigned integer value.
 */
static uint32_t read_u32_le(std::ifstream& f) { uint32_t v; f.read(reinterpret_cast<char*>(&v), 4); return v; }

/**
 * @brief Reads a 16-bit unsigned integer from the file stream in little-endian format.
 * @param f The input file stream.
 * @return The 16-bit unsigned integer value.
 */
static uint16_t read_u16_le(std::ifstream& f) { uint16_t v; f.read(reinterpret_cast<char*>(&v), 2); return v; }

/**
 * @brief Loads a WAV file from the specified path into an AudioBuffer object.
 *
 * This function handles parsing the RIFF and WAVE headers, locating the 'fmt ' and 'data' chunks,
 * and decoding various PCM and floating-point audio formats (16-bit PCM, 24-bit PCM, 32-bit Float)
 * into the internal float-based AudioBuffer format.
 *
 * @param path The file system path to the WAV file.
 * @return A core::AudioBuffer containing the loaded audio data, normalized to [-1.0, 1.0].
 * @throw std::runtime_error if the file cannot be opened, is not a valid RIFF/WAV file,
 * or contains an unsupported audio format.
 */
core::AudioBuffer AudioLoader::loadWav(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open WAV: " + path);

    // RIFF header
    char riff[4]; f.read(riff, 4);
    if (f.gcount() != 4 || std::string(riff, 4) != "RIFF") throw std::runtime_error("Not a RIFF file");
    // Skip chunk size
    (void)read_u32_le(f);
    char wave[4]; f.read(wave, 4);
    if (f.gcount() != 4 || std::string(wave, 4) != "WAVE") throw std::runtime_error("Not a WAVE file");

    uint16_t audioFormat = 0, numChannels = 0, bitsPerSample = 0;
    uint32_t sampleRate = 0, dataSize = 0;
    std::streampos dataPos = 0;

    // Chunk reading loop
    while (f && !f.eof()) {
        char id[4]; f.read(id, 4);
        if (f.gcount() != 4) break;
        uint32_t chunkSize = read_u32_le(f);
        std::string chunkId(id, 4);
        if (chunkId == "fmt ") {
            audioFormat = read_u16_le(f);
            numChannels = read_u16_le(f);
            sampleRate = read_u32_le(f);
            // Skip ByteRate and BlockAlign
            (void)read_u32_le(f); (void)read_u16_le(f);
            bitsPerSample = read_u16_le(f);
            // Skip any remaining 'fmt ' chunk extensions
            if (chunkSize > 16) f.seekg(chunkSize - 16, std::ios::cur);
        } else if (chunkId == "data") {
            dataSize = chunkSize;
            // Store the position of the start of the data chunk
            dataPos = f.tellg();
            f.seekg(chunkSize, std::ios::cur);
        } else {
            // Skip unknown chunk
            f.seekg(chunkSize, std::ios::cur);
        }
        // Handle chunk padding (if chunkSize is odd)
        if (chunkSize % 2 == 1) f.seekg(1, std::ios::cur);
    }

    if (audioFormat == 0 || numChannels == 0 || dataSize == 0) throw std::runtime_error("Invalid WAV: Missing 'fmt ' or 'data' chunks");

    // Calculate total number of samples and frames
    size_t totalSamples = dataSize * 8ull / bitsPerSample;
    size_t frames = totalSamples / numChannels;
    core::AudioBuffer buffer(numChannels, frames, static_cast<float>(sampleRate));

    // Rewind file stream to the start of the data chunk
    f.clear(); f.seekg(dataPos);

    // Lambda for decoding 16-bit signed PCM
    auto decodePCM16 = [&](void) {
        for (size_t i = 0; i < frames; ++i) {
            for (size_t ch = 0; ch < numChannels; ++ch) {
                int16_t s; f.read(reinterpret_cast<char*>(&s), 2);
                // Normalize to [-1.0, 1.0]
                buffer.getChannel(ch)[i] = static_cast<float>(s) / 32768.0f;
            }
        }
    };
    // Lambda for decoding 24-bit signed PCM (stored in 3 bytes, little-endian)
    auto decodePCM24 = [&](void) {
        for (size_t i = 0; i < frames; ++i) {
            for (size_t ch = 0; ch < numChannels; ++ch) {
                uint8_t b[3]; f.read(reinterpret_cast<char*>(b), 3);
                // Reconstruct the 24-bit signed integer
                int32_t v = (b[0]) | (b[1] << 8) | (b[2] << 16);
                // Sign extension for 24-bit value
                if (v & 0x800000) v |= ~0xFFFFFF;
                // Normalize to [-1.0, 1.0]
                buffer.getChannel(ch)[i] = static_cast<float>(v) / 8388608.0f;
            }
        }
    };
    // Lambda for decoding 32-bit floating-point audio
    auto decodeFloat32 = [&](void) {
        for (size_t i = 0; i < frames; ++i) {
            for (size_t ch = 0; ch < numChannels; ++ch) {
                char bytes[4];
                f.read(bytes, 4);
                float s;
                // Use memcpy to safely read bytes into a float
                std::memcpy(&s, bytes, 4);
                buffer.getChannel(ch)[i] = s;
            }
        }
    };

    // Dispatch decoding based on format and bit depth
    if (audioFormat == 1) { // PCM
        if (bitsPerSample == 16) decodePCM16();
        else if (bitsPerSample == 24) decodePCM24();
        else throw std::runtime_error("Unsupported PCM bit depth: " + std::to_string(bitsPerSample));
    } else if (audioFormat == 3 && bitsPerSample == 32) { // IEEE Float
        decodeFloat32();
    } else {
        throw std::runtime_error("Unsupported audio format or bit depth: Format=" + std::to_string(audioFormat) + ", Bits=" + std::to_string(bitsPerSample));
    }
    return buffer;
}

} // namespace ave::pipeline