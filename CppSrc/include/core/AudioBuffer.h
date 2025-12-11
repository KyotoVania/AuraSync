#pragma once

#include <vector>
#include <memory>
#include <cstring>

// Pre-define the value of Pi
#define M_PI 3.14159265358979323846


namespace ave::core {

/**
 * @brief Multi-channel audio buffer with metadata.
 *
 * This class supports both interleaved and non-interleaved data layouts.
 */
class AudioBuffer {
public:
    /**
     * @brief Defines the data layout of the audio buffer.
     */
    enum class Layout {
        INTERLEAVED,      ///< Samples are ordered L R L R L R...
        NON_INTERLEAVED   ///< Samples are ordered L L L... R R R...
    };

    /**
     * @brief Default constructor.
     */
    AudioBuffer() = default;

    /**
     * @brief Constructs an AudioBuffer with specified dimensions.
     *
     * @param channels The number of audio channels.
     * @param frames The number of sample frames per channel.
     * @param sampleRate The sample rate of the audio data in Hz (default is 44100.0f).
     */
    AudioBuffer(size_t channels, size_t frames, float sampleRate = 44100.0f);

    // Data access
    /**
     * @brief Gets a pointer to the start of the specified channel's data.
     *
     * @param channel The index of the channel (0 to getChannelCount() - 1).
     * @return A pointer to the channel's float data.
     */
    float* getChannel(size_t channel);

    /**
     * @brief Gets a const pointer to the start of the specified channel's data.
     *
     * @param channel The index of the channel (0 to getChannelCount() - 1).
     * @return A const pointer to the channel's float data.
     */
    const float* getChannel(size_t channel) const;

    /**
     * @brief Creates and returns a mono mixdown of the buffer.
     *
     * The resulting vector contains the average of all channels for each frame.
     * @return A vector of floats representing the mono mix.
     */
    std::vector<float> getMono() const;

    // Properties
    /**
     * @brief Returns the number of channels in the buffer.
     * @return The channel count.
     */
    size_t getChannelCount() const { return m_channels; }

    /**
     * @brief Returns the number of frames (samples per channel) in the buffer.
     * @return The frame count.
     */
    size_t getFrameCount() const { return m_frames; }

    /**
     * @brief Returns the sample rate of the audio data in Hz.
     * @return The sample rate.
     */
    float getSampleRate() const { return m_sampleRate; }

    /**
     * @brief Calculates the total duration of the buffer in seconds.
     * @return The duration in seconds.
     */
    double getDuration() const { return m_frames / static_cast<double>(m_sampleRate); }

    /**
     * @brief Returns the memory layout of the buffer (Interleaved or Non-Interleaved).
     * @return The buffer layout.
     */
    Layout getLayout() const { return m_layout; }

    // Operations
    /**
     * @brief Resizes the audio buffer, retaining existing data if possible.
     *
     * The internal data vector is resized and metadata is updated.
     * @param channels The new number of channels.
     * @param frames The new number of frames.
     */
    void resize(size_t channels, size_t frames);

    /**
     * @brief Clears (sets to zero) all sample data in the buffer.
     */
    void clear();

    /**
     * @brief Creates a deep copy of the current AudioBuffer.
     * @return A new AudioBuffer object with copied data.
     */
    AudioBuffer clone() const;

    // Slicing
    /**
     * @brief Creates a new AudioBuffer containing a slice of the current buffer's data.
     *
     * @param startFrame The starting frame index (inclusive).
     * @param endFrame The ending frame index (exclusive).
     * @return A new AudioBuffer containing the slice.
     */
    AudioBuffer slice(size_t startFrame, size_t endFrame) const;

    // Raw data access (for FFT etc.)
    /**
     * @brief Gets a pointer to the raw underlying data vector.
     * @return A pointer to the raw float data.
     */
    float* data() { return m_data.data(); }

    /**
     * @brief Gets a const pointer to the raw underlying data vector.
     * @return A const pointer to the raw float data.
     */
    const float* data() const { return m_data.data(); }

    /**
     * @brief Returns the total number of floats in the underlying data vector.
     * @return The size of the raw data vector.
     */
    size_t dataSize() const { return m_data.size(); }

private:
    std::vector<float> m_data;
    size_t m_channels = 0;
    size_t m_frames = 0;
    float m_sampleRate = 44100.0f;
    Layout m_layout = Layout::NON_INTERLEAVED;

    /**
     * @brief Calculates the starting index for a specific channel within the raw data vector.
     * @param channel The channel index.
     * @return The offset index.
     */
    size_t channelOffset(size_t channel) const;
};

/**
 * @brief Represents a single spectral frame, typically resulting from an STFT or FFT operation.
 *
 * It stores magnitude and phase information for a moment in time.
 */
class SpectralFrame {
public:
    std::vector<float> magnitudes;    ///< Magnitude spectrum (bin size is usually FFT size / 2 + 1)
    std::vector<float> phases;        ///< Phase spectrum
    float timestamp = 0.0f;           ///< Time in seconds corresponding to this frame's center

    /**
     * @brief Calculates the total energy within a specified frequency band.
     *
     * @param freqLow The lower frequency bound (in Hz).
     * @param freqHigh The upper frequency bound (in Hz).
     * @param sampleRate The sample rate used for the original audio signal.
     * @return The summed energy in the band.
     */
    float getBandEnergy(float freqLow, float freqHigh, float sampleRate) const;

    // Spectral features
    /**
     * @brief Calculates the Spectral Centroid, a measure of the spectral shape.
     * @return The spectral centroid value.
     */
    float getSpectralCentroid() const;

    /**
     * @brief Calculates the Spectral Flux between the current and a previous frame.
     *
     * Spectral Flux measures the change in the magnitude spectrum.
     * @param previous The preceding SpectralFrame.
     * @return The calculated spectral flux value.
     */
    float getSpectralFlux(const SpectralFrame& previous) const;

    /**
     * @brief Calculates the Spectral Rolloff, the frequency below which a specified percentage of the total energy lies.
     *
     * @param threshold The energy percentage threshold (e.g., 0.85 for 85%).
     * @return The spectral rolloff frequency (in Hz).
     */
    float getSpectralRolloff(float threshold = 0.85f) const;
};

/**
 * @brief Utility namespace for generating and applying window functions for spectral analysis.
 */
namespace window {

    /**
     * @brief Generates a Hann (Hanning) window function.
     * @param size The size (number of points) of the window.
     * @return A vector containing the window coefficients.
     */
    std::vector<float> hann(size_t size);

    /**
     * @brief Generates a Hamming window function.
     * @param size The size (number of points) of the window.
     * @return A vector containing the window coefficients.
     */
    std::vector<float> hamming(size_t size);

    /**
     * @brief Generates a Blackman window function.
     * @param size The size (number of points) of the window.
     * @return A vector containing the window coefficients.
     */
    std::vector<float> blackman(size_t size);

    /**
     * @brief Applies a window function in-place to an array of audio data.
     *
     * The output data is multiplied element-wise by the window coefficients.
     * @param data Pointer to the float array to be windowed (will be modified).
     * @param window Pointer to the float array of window coefficients.
     * @param size The number of elements in the data and window arrays.
     */
    void apply(float* data, const float* window, size_t size);
}

} // namespace ave::core