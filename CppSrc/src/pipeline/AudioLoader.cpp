#include "../../include/pipeline/AudioLoader.h"
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>

namespace ave::pipeline {

static uint32_t read_u32_le(std::ifstream& f) { uint32_t v; f.read(reinterpret_cast<char*>(&v), 4); return v; }
static uint16_t read_u16_le(std::ifstream& f) { uint16_t v; f.read(reinterpret_cast<char*>(&v), 2); return v; }

core::AudioBuffer AudioLoader::loadWav(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open WAV: " + path);

    // RIFF header
    char riff[4]; f.read(riff, 4);
    if (f.gcount() != 4 || std::string(riff, 4) != "RIFF") throw std::runtime_error("Not a RIFF file");
    (void)read_u32_le(f); // file size
    char wave[4]; f.read(wave, 4);
    if (f.gcount() != 4 || std::string(wave, 4) != "WAVE") throw std::runtime_error("Not a WAVE file");

    // Parse chunks
    uint16_t audioFormat = 0; // 1=PCM, 3=IEEE float
    uint16_t numChannels = 0;
    uint32_t sampleRate = 0;
    uint16_t bitsPerSample = 0;
    uint32_t dataSize = 0;
    std::streampos dataPos = 0;

    while (f && !f.eof()) {
        char id[4]; f.read(id, 4);
        if (f.gcount() != 4) break;
        uint32_t chunkSize = read_u32_le(f);
        std::string chunkId(id, 4);
        if (chunkId == "fmt ") {
            audioFormat = read_u16_le(f);
            numChannels = read_u16_le(f);
            sampleRate = read_u32_le(f);
            (void)read_u32_le(f); // byte rate
            (void)read_u16_le(f); // block align
            bitsPerSample = read_u16_le(f);
            // Skip any extension bytes
            if (chunkSize > 16) {
                f.seekg(chunkSize - 16, std::ios::cur);
            }
        } else if (chunkId == "data") {
            dataSize = chunkSize;
            dataPos = f.tellg();
            f.seekg(chunkSize, std::ios::cur);
        } else {
            // skip unknown chunk
            f.seekg(chunkSize, std::ios::cur);
        }
        // Chunks are word-aligned; if odd size, skip pad byte
        if (chunkSize % 2 == 1) f.seekg(1, std::ios::cur);
    }

    if (audioFormat == 0 || numChannels == 0 || sampleRate == 0 || bitsPerSample == 0 || dataSize == 0) {
        throw std::runtime_error("Invalid or unsupported WAV file");
    }

    // Prepare buffer
    size_t totalSamples = dataSize * 8ull / bitsPerSample; // across all channels
    size_t frames = static_cast<size_t>(totalSamples / numChannels);
    core::AudioBuffer buffer(numChannels, frames, static_cast<float>(sampleRate));

    // Read and decode samples
    f.clear();
    f.seekg(dataPos);

    auto decodePCM16 = [&](void) {
        for (size_t i = 0; i < frames; ++i) {
            for (size_t ch = 0; ch < numChannels; ++ch) {
                int16_t s; f.read(reinterpret_cast<char*>(&s), 2);
                buffer.getChannel(ch)[i] = static_cast<float>(s) / 32768.0f;
            }
        }
    };

    auto decodePCM24 = [&](void) {
        for (size_t i = 0; i < frames; ++i) {
            for (size_t ch = 0; ch < numChannels; ++ch) {
                uint8_t b[3]; f.read(reinterpret_cast<char*>(b), 3);
                int32_t v = (b[0]) | (b[1] << 8) | (b[2] << 16);
                // sign-extend 24-bit
                if (v & 0x800000) v |= ~0xFFFFFF;
                buffer.getChannel(ch)[i] = static_cast<float>(v) / 8388608.0f; // 2^23
            }
        }
    };

    auto decodeFloat32 = [&](void) {
        for (size_t i = 0; i < frames; ++i) {
            for (size_t ch = 0; ch < numChannels; ++ch) {
                float s; f.read(reinterpret_cast<char*>(&s), 4);
                buffer.getChannel(ch)[i] = s;
            }
        }
    };

    if (audioFormat == 1) {
        if (bitsPerSample == 16) decodePCM16();
        else if (bitsPerSample == 24) decodePCM24();
        else throw std::runtime_error("Unsupported PCM bit depth: " + std::to_string(bitsPerSample));
    } else if (audioFormat == 3) {
        if (bitsPerSample == 32) decodeFloat32();
        else throw std::runtime_error("Unsupported IEEE float bit depth: " + std::to_string(bitsPerSample));
    } else {
        throw std::runtime_error("Unsupported WAV format code: " + std::to_string(audioFormat));
    }

    return buffer;
}

} // namespace ave::pipeline

