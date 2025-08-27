# Python script pour générer un métronome WAV
import numpy as np
import wave

def create_metronome(bpm, duration_sec, filename):
    sample_rate = 44100
    samples = int(sample_rate * duration_sec)
    signal = np.zeros(samples)

    # Intervalle entre les clicks en échantillons
    interval = int(sample_rate * 60 / bpm)

    # Génération des clicks
    for i in range(0, samples, interval):
        # Click court (10ms)
        click_duration = int(0.01 * sample_rate)
        for j in range(min(click_duration, samples - i)):
            signal[i + j] = 0.8 * np.sin(2 * np.pi * 1000 * j / sample_rate)

    # Sauvegarde en WAV
    signal = (signal * 32767).astype(np.int16)
    with wave.open(filename, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(signal.tobytes())

# Créer un métronome à 120 BPM
create_metronome(130, 10, "../CppSrc/tests/assets/test_130bpm.wav")