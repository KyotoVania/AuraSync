#include "../../include/core/AudioBuffer.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace ave::core {

// ============================================================================
// AudioBuffer Implementation
// ============================================================================

/**
 * @brief Constructs an AudioBuffer with specified dimensions.
 *
 * Initializes the buffer data and sets metadata (channels, frames, sample rate).
 * The internal data vector is resized and zeroed out.
 * @param channels The number of audio channels.
 * @param frames The number of sample frames per channel.
 * @param sampleRate The sample rate of the audio data in Hz.
 */
AudioBuffer::AudioBuffer(size_t channels, size_t frames, float sampleRate)
    : m_channels(channels)
    , m_frames(frames)
    , m_sampleRate(sampleRate)
    , m_layout(Layout::NON_INTERLEAVED) {
    resize(channels, frames);
}

/**
 * @brief Gets a pointer to the start of the specified channel's data.
 *
 * @param channel The index of the channel (0 to getChannelCount() - 1).
 * @return A pointer to the channel's float data, or nullptr if the channel index is out of bounds.
 */
float* AudioBuffer::getChannel(size_t channel) {
    if (channel >= m_channels) {
        return nullptr;
    }
    return m_data.data() + channelOffset(channel);
}

/**
 * @brief Gets a const pointer to the start of the specified channel's data.
 *
 * @param channel The index of the channel (0 to getChannelCount() - 1).
 * @return A const pointer to the channel's float data, or nullptr if the channel index is out of bounds.
 */
const float* AudioBuffer::getChannel(size_t channel) const {
    if (channel >= m_channels) {
        return nullptr;
    }
    return m_data.data() + channelOffset(channel);
}

/**
 * @brief Creates and returns a mono mixdown of the buffer.
 *
 * The resulting vector contains the average of all channels for each frame.
 * @return A vector of floats representing the mono mix. Returns an empty vector if the buffer is empty.
 */
std::vector<float> AudioBuffer::getMono() const {
    if (m_channels == 0 || m_frames == 0) {
        return {};
    }

    std::vector<float> mono(m_frames, 0.0f);

    if (m_channels == 1) {
        // Already mono, just copy the channel data
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

        // Normalize by dividing by the number of channels
        float scale = 1.0f / m_channels;
        for (float& sample : mono) {
            sample *= scale;
        }
    }

    return mono;
}

/**
 * @brief Resizes the audio buffer, retaining existing data if possible.
 *
 * The internal data vector is resized and metadata is updated. The newly allocated space is zeroed out.
 * @param channels The new number of channels.
 * @param frames The new number of frames.
 */
void AudioBuffer::resize(size_t channels, size_t frames) {
    m_channels = channels;
    m_frames = frames;
    // Resize the underlying data storage and initialize new elements to zero
    m_data.resize(channels * frames);
    std::fill(m_data.begin(), m_data.end(), 0.0f);
}

/**
 * @brief Clears (sets to zero) all sample data in the buffer.
 */
void AudioBuffer::clear() {
    std::fill(m_data.begin(), m_data.end(), 0.0f);
}

/**
 * @brief Creates a deep copy of the current AudioBuffer.
 * @return A new AudioBuffer object with copied data.
 */
AudioBuffer AudioBuffer::clone() const {
    AudioBuffer copy(m_channels, m_frames, m_sampleRate);
    copy.m_data = m_data;
    copy.m_layout = m_layout;
    return copy;
}

/**
 * @brief Creates a new AudioBuffer containing a slice of the current buffer's data.
 *
 * @param startFrame The starting frame index (inclusive).
 * @param endFrame The ending frame index (exclusive).
 * @return A new AudioBuffer containing the slice. Returns an empty buffer if slicing parameters are invalid.
 */
AudioBuffer AudioBuffer::slice(size_t startFrame, size_t endFrame) const {
    if (startFrame >= m_frames || endFrame > m_frames || startFrame >= endFrame) {
        return AudioBuffer(); // Return an empty buffer if parameters are invalid
    }

    size_t sliceFrames = endFrame - startFrame;
    AudioBuffer sliced(m_channels, sliceFrames, m_sampleRate);

    // Copy data channel by channel
    for (size_t ch = 0; ch < m_channels; ++ch) {
        const float* srcChannel = getChannel(ch) + startFrame;
        float* dstChannel = sliced.getChannel(ch);
        std::copy(srcChannel, srcChannel + sliceFrames, dstChannel);
    }

    return sliced;
}

/**
 * @brief Calculates the starting index for a specific channel within the raw data vector.
 * @param channel The channel index.
 * @return The offset index based on the current layout.
 */
size_t AudioBuffer::channelOffset(size_t channel) const {
    if (m_layout == Layout::NON_INTERLEAVED) {
        // In non-interleaved (planar) layout, channels are sequential blocks: LLL...RRR...
        return channel * m_frames;
    } else {
        // In interleaved layout, samples are LRLR...
        // The start of a channel's data is simply its index, but subsequent samples
        // are accessed by multiplying the frame index by the channel count.
        // NOTE: This implementation only supports NON_INTERLEAVED for now for full array access.
        return channel;
    }
}

// ============================================================================
// SpectralFrame Implementation
// ============================================================================

/**
 * @brief Calculates the total energy within a specified frequency band.
 *
 * @param freqLow The lower frequency bound (in Hz).
 * @param freqHigh The upper frequency bound (in Hz).
 * @param sampleRate The sample rate used for the original audio signal.
 * @return The summed energy (squared magnitudes) in the band. Returns 0.0f if magnitudes are empty.
 */
float SpectralFrame::getBandEnergy(float freqLow, float freqHigh, float sampleRate) const {
    if (magnitudes.empty()) {
        return 0.0f;
    }

    // Bin frequency resolution: sampleRate / FFT size. Since magnitude vector size is N/2+1 (half-spectrum),
    // the max frequency (Nyquist) is at index N/2, corresponding to magnitude.size()-1.
    // The total frequency span is SampleRate / 2, covered by magnitude.size() bins (from 0 to N/2).
    float binFreq = (sampleRate / 2.0f) / static_cast<float>(magnitudes.size() - 1);

    // Convert frequencies to bin indices
    size_t binLow = static_cast<size_t>(freqLow / binFreq);
    size_t binHigh = static_cast<size_t>(freqHigh / binFreq);

    // Clamp indices to valid range
    binLow = std::min(binLow, magnitudes.size() - 1);
    binHigh = std::min(binHigh, magnitudes.size() - 1);

    // Sum energy (magnitude squared) in band
    float energy = 0.0f;
    for (size_t i = binLow; i <= binHigh; ++i) {
        energy += magnitudes[i] * magnitudes[i];
    }

    return energy;
}

/**
 * @brief Calculates the Spectral Centroid, a measure of the spectral shape.
 *
 * The centroid represents the "center of gravity" of the spectrum, with higher values indicating "brighter" sounds.
 * @return The spectral centroid value (in bin units). Returns 0.0f if the spectrum is empty or silent.
 */
float SpectralFrame::getSpectralCentroid() const {
    if (magnitudes.empty()) {
        return 0.0f;
    }

    float weightedSum = 0.0f;
    float magnitudeSum = 0.0f;

    for (size_t i = 0; i < magnitudes.size(); ++i) {
        // Weighted sum is sum(index * magnitude)
        weightedSum += static_cast<float>(i) * magnitudes[i];
        // Total magnitude sum is sum(magnitude)
        magnitudeSum += magnitudes[i];
    }

    if (magnitudeSum == 0.0f) {
        return 0.0f;
    }

    // Centroid = Weighted Sum / Total Sum
    return weightedSum / magnitudeSum;
}

/**
 * @brief Calculates the Spectral Flux between the current and a previous frame.
 *
 * Spectral Flux measures the change in the magnitude spectrum, often used for onset detection.
 * Only positive differences (increases in magnitude) contribute (half-wave rectification).
 * @param previous The preceding SpectralFrame.
 * @return The calculated spectral flux value. Returns 0.0f if the frame sizes do not match.
 */
float SpectralFrame::getSpectralFlux(const SpectralFrame& previous) const {
    if (magnitudes.size() != previous.magnitudes.size()) {
        return 0.0f;
    }

    float flux = 0.0f;
    for (size_t i = 0; i < magnitudes.size(); ++i) {
        float diff = magnitudes[i] - previous.magnitudes[i];
        if (diff > 0.0f) { // Only count positive differences (increase in energy)
            flux += diff;
        }
    }

    return flux;
}

/**
 * @brief Calculates the Spectral Rolloff, the frequency below which a specified percentage of the total energy lies.
 *
 * @param threshold The energy percentage threshold (e.g., 0.85 for 85%).
 * @return The spectral rolloff frequency normalized to the range [0.0, 1.0], where 1.0 represents the Nyquist frequency. Returns 1.0f if the threshold is never reached.
 */
float SpectralFrame::getSpectralRolloff(float threshold) const {
    if (magnitudes.empty()) {
        return 0.0f;
    }

    // Calculate total energy (magnitude squared)
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
            // Normalize the bin index (i) by the total number of bins (magnitudes.size() - 1, but using size for simplicity/approximation)
            return static_cast<float>(i) / static_cast<float>(magnitudes.size());
        }
    }

    // If the threshold is never reached (e.g., threshold=1.0 and totalEnergy is non-zero)
    return 1.0f;
}

// ============================================================================
// Window Functions Implementation
// ============================================================================

namespace window {

/**
 * @brief Generates a Hann (Hanning) window function.
 * @param size The size (number of points) of the window.
 * @return A vector containing the window coefficients.
 */
std::vector<float> hann(size_t size) {
    std::vector<float> window(size);
    // Hann window formula: 0.5 * (1 - cos(2*pi*i / (N-1)))
    for (size_t i = 0; i < size; ++i) {
        float t = static_cast<float>(i) / (size - 1);
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * t));
    }
    return window;
}

/**
 * @brief Generates a Hamming window function.
 * @param size The size (number of points) of the window.
 * @return A vector containing the window coefficients.
 */
std::vector<float> hamming(size_t size) {
    std::vector<float> window(size);
    // Hamming window formula: 0.54 - 0.46 * cos(2*pi*i / (N-1))
    for (size_t i = 0; i < size; ++i) {
        float t = static_cast<float>(i) / (size - 1);
        window[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * t);
    }
    return window;
}

/**
 * @brief Generates a Blackman window function.
 * @param size The size (number of points) of the window.
 * @return A vector containing the window coefficients.
 */
std::vector<float> blackman(size_t size) {
    std::vector<float> window(size);
    // Blackman window formula: 0.42 - 0.5*cos(2*pi*i/(N-1)) + 0.08*cos(4*pi*i/(N-1))
    for (size_t i = 0; i < size; ++i) {
        float t = static_cast<float>(i) / (size - 1);
        window[i] = 0.42f - 0.5f * std::cos(2.0f * M_PI * t)
                    + 0.08f * std::cos(4.0f * M_PI * t);
    }
    return window;
}

/**
 * @brief Applies a window function in-place to an array of audio data.
 *
 * The output data is multiplied element-wise by the window coefficients.
 * @param data Pointer to the float array to be windowed (will be modified).
 * @param window Pointer to the float array of window coefficients.
 * @param size The number of elements in the data and window arrays.
 */
void apply(float* data, const float* window, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] *= window[i];
    }
}

} // namespace window

} // namespace ave::core