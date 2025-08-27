#include "../../include/core/AudioBuffer.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace ave::core {

// ============================================================================
// AudioBuffer Implementation
// ============================================================================

AudioBuffer::AudioBuffer(size_t channels, size_t frames, float sampleRate)
    : m_channels(channels)
    , m_frames(frames)
    , m_sampleRate(sampleRate)
    , m_layout(Layout::NON_INTERLEAVED) {
    resize(channels, frames);
}

float* AudioBuffer::getChannel(size_t channel) {
    if (channel >= m_channels) {
        return nullptr;
    }
    return m_data.data() + channelOffset(channel);
}

const float* AudioBuffer::getChannel(size_t channel) const {
    if (channel >= m_channels) {
        return nullptr;
    }
    return m_data.data() + channelOffset(channel);
}

std::vector<float> AudioBuffer::getMono() const {
    if (m_channels == 0 || m_frames == 0) {
        return {};
    }
    
    std::vector<float> mono(m_frames, 0.0f);
    
    if (m_channels == 1) {
        // Already mono
        const float* channel = getChannel(0);
        std::copy(channel, channel + m_frames, mono.begin());
    } else {
        // Mix all channels
        for (size_t ch = 0; ch < m_channels; ++ch) {
            const float* channel = getChannel(ch);
            for (size_t i = 0; i < m_frames; ++i) {
                mono[i] += channel[i];
            }
        }
        
        // Normalize
        float scale = 1.0f / m_channels;
        for (float& sample : mono) {
            sample *= scale;
        }
    }
    
    return mono;
}

void AudioBuffer::resize(size_t channels, size_t frames) {
    m_channels = channels;
    m_frames = frames;
    m_data.resize(channels * frames);
    std::fill(m_data.begin(), m_data.end(), 0.0f);
}

void AudioBuffer::clear() {
    std::fill(m_data.begin(), m_data.end(), 0.0f);
}

AudioBuffer AudioBuffer::clone() const {
    AudioBuffer copy(m_channels, m_frames, m_sampleRate);
    copy.m_data = m_data;
    copy.m_layout = m_layout;
    return copy;
}

AudioBuffer AudioBuffer::slice(size_t startFrame, size_t endFrame) const {
    if (startFrame >= m_frames || endFrame > m_frames || startFrame >= endFrame) {
        return AudioBuffer(); // Empty buffer
    }
    
    size_t sliceFrames = endFrame - startFrame;
    AudioBuffer sliced(m_channels, sliceFrames, m_sampleRate);
    
    for (size_t ch = 0; ch < m_channels; ++ch) {
        const float* srcChannel = getChannel(ch) + startFrame;
        float* dstChannel = sliced.getChannel(ch);
        std::copy(srcChannel, srcChannel + sliceFrames, dstChannel);
    }
    
    return sliced;
}

size_t AudioBuffer::channelOffset(size_t channel) const {
    if (m_layout == Layout::NON_INTERLEAVED) {
        return channel * m_frames;
    } else {
        // Interleaved layout would start at channel index
        return channel;
    }
}

// ============================================================================
// SpectralFrame Implementation
// ============================================================================

float SpectralFrame::getBandEnergy(float freqLow, float freqHigh, float sampleRate) const {
    if (magnitudes.empty()) {
        return 0.0f;
    }
    
    // Convert frequencies to bin indices
    float binFreq = sampleRate / (2.0f * magnitudes.size());
    size_t binLow = static_cast<size_t>(freqLow / binFreq);
    size_t binHigh = static_cast<size_t>(freqHigh / binFreq);
    
    binLow = std::min(binLow, magnitudes.size() - 1);
    binHigh = std::min(binHigh, magnitudes.size() - 1);
    
    // Sum energy in band
    float energy = 0.0f;
    for (size_t i = binLow; i <= binHigh; ++i) {
        energy += magnitudes[i] * magnitudes[i];
    }
    
    return energy;
}

float SpectralFrame::getSpectralCentroid() const {
    if (magnitudes.empty()) {
        return 0.0f;
    }
    
    float weightedSum = 0.0f;
    float magnitudeSum = 0.0f;
    
    for (size_t i = 0; i < magnitudes.size(); ++i) {
        weightedSum += i * magnitudes[i];
        magnitudeSum += magnitudes[i];
    }
    
    if (magnitudeSum == 0.0f) {
        return 0.0f;
    }
    
    return weightedSum / magnitudeSum;
}

float SpectralFrame::getSpectralFlux(const SpectralFrame& previous) const {
    if (magnitudes.size() != previous.magnitudes.size()) {
        return 0.0f;
    }
    
    float flux = 0.0f;
    for (size_t i = 0; i < magnitudes.size(); ++i) {
        float diff = magnitudes[i] - previous.magnitudes[i];
        if (diff > 0.0f) { // Half-wave rectification
            flux += diff;
        }
    }
    
    return flux;
}

float SpectralFrame::getSpectralRolloff(float threshold) const {
    if (magnitudes.empty()) {
        return 0.0f;
    }
    
    // Calculate total energy
    float totalEnergy = 0.0f;
    for (float mag : magnitudes) {
        totalEnergy += mag * mag;
    }
    
    if (totalEnergy == 0.0f) {
        return 0.0f;
    }
    
    // Find rolloff point
    float cumulativeEnergy = 0.0f;
    float targetEnergy = totalEnergy * threshold;
    
    for (size_t i = 0; i < magnitudes.size(); ++i) {
        cumulativeEnergy += magnitudes[i] * magnitudes[i];
        if (cumulativeEnergy >= targetEnergy) {
            return static_cast<float>(i) / magnitudes.size();
        }
    }
    
    return 1.0f;
}

// ============================================================================
// Window Functions Implementation
// ============================================================================

namespace window {

std::vector<float> hann(size_t size) {
    std::vector<float> window(size);
    for (size_t i = 0; i < size; ++i) {
        float t = static_cast<float>(i) / (size - 1);
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * t));
    }
    return window;
}

std::vector<float> hamming(size_t size) {
    std::vector<float> window(size);
    for (size_t i = 0; i < size; ++i) {
        float t = static_cast<float>(i) / (size - 1);
        window[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * t);
    }
    return window;
}

std::vector<float> blackman(size_t size) {
    std::vector<float> window(size);
    for (size_t i = 0; i < size; ++i) {
        float t = static_cast<float>(i) / (size - 1);
        window[i] = 0.42f - 0.5f * std::cos(2.0f * M_PI * t) 
                    + 0.08f * std::cos(4.0f * M_PI * t);
    }
    return window;
}

void apply(float* data, const float* window, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] *= window[i];
    }
}

} // namespace window

} // namespace ave::core