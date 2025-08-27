#pragma once

#include <vector>
#include <memory>
#include <cstring>

//define M_PI 3.14159265358979323846
#define M_PI 3.14159265358979323846


namespace ave::core {

/**
 * Multi-channel audio buffer with metadata
 * Supports both interleaved and non-interleaved layouts
 */
class AudioBuffer {
public:
    enum class Layout {
        INTERLEAVED,      // L R L R L R...
        NON_INTERLEAVED   // L L L... R R R...
    };

    // Constructors
    AudioBuffer() = default;
    AudioBuffer(size_t channels, size_t frames, float sampleRate = 44100.0f);
    
    // Data access
    float* getChannel(size_t channel);
    const float* getChannel(size_t channel) const;
    
    // Get mono mix (creates temporary buffer)
    std::vector<float> getMono() const;
    
    // Properties
    size_t getChannelCount() const { return m_channels; }
    size_t getFrameCount() const { return m_frames; }
    float getSampleRate() const { return m_sampleRate; }
    double getDuration() const { return m_frames / static_cast<double>(m_sampleRate); }
    Layout getLayout() const { return m_layout; }
    
    // Operations
    void resize(size_t channels, size_t frames);
    void clear();
    AudioBuffer clone() const;
    
    // Slicing
    AudioBuffer slice(size_t startFrame, size_t endFrame) const;
    
    // Raw data access (for FFT etc.)
    float* data() { return m_data.data(); }
    const float* data() const { return m_data.data(); }
    size_t dataSize() const { return m_data.size(); }

private:
    std::vector<float> m_data;
    size_t m_channels = 0;
    size_t m_frames = 0;
    float m_sampleRate = 44100.0f;
    Layout m_layout = Layout::NON_INTERLEAVED;
    
    size_t channelOffset(size_t channel) const;
};

/**
 * Helper for STFT/FFT operations
 */
class SpectralFrame {
public:
    std::vector<float> magnitudes;    // Magnitude spectrum
    std::vector<float> phases;        // Phase spectrum  
    float timestamp = 0.0f;           // Time in seconds
    
    // Frequency band extraction
    float getBandEnergy(float freqLow, float freqHigh, float sampleRate) const;
    
    // Spectral features
    float getSpectralCentroid() const;
    float getSpectralFlux(const SpectralFrame& previous) const;
    float getSpectralRolloff(float threshold = 0.85f) const;
};

/**
 * Window functions for spectral analysis
 */
namespace window {
    std::vector<float> hann(size_t size);
    std::vector<float> hamming(size_t size);
    std::vector<float> blackman(size_t size);
    void apply(float* data, const float* window, size_t size);
}

} // namespace ave::core