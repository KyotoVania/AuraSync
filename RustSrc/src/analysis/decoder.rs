use anyhow::{Context, Result};
use hound::{SampleFormat, WavReader};
use rubato::{FftFixedIn, Resampler};
use std::path::PathBuf;

/// Charge et décode un fichier WAV en conservant les canaux séparés (stéréo) avec rééchantillonnage à target_sr
pub fn decode_to_channels(path: &PathBuf, target_sr: u32) -> Result<Vec<Vec<f32>>> {
    let mut reader = WavReader::open(path)
        .with_context(|| format!("Impossible d'ouvrir le fichier {:?}", path))?;

    let spec = reader.spec();
    let channels = spec.channels as usize;
    let source_sr = spec.sample_rate;

    // Lecture des échantillons interleavés en f32 normalisé
    let interleaved: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => {
            reader.samples::<f32>()
                .collect::<Result<Vec<_>, _>>()
                .context("Erreur lors de la lecture des échantillons")?
        }
        SampleFormat::Int => {
            let bit_depth = spec.bits_per_sample;
            let max_val = (1 << (bit_depth - 1)) as f32;
            reader.samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<Result<Vec<_>, _>>()
                .context("Erreur lors de la conversion des échantillons")?
        }
    };

    // Séparation des canaux
    let frames = interleaved.len() / channels.max(1);
    let mut per_channel: Vec<Vec<f32>> = (0..channels).map(|_| Vec::with_capacity(frames)).collect();
    if channels > 1 {
        for frame in interleaved.chunks(channels) {
            for (ch, sample) in frame.iter().enumerate() { per_channel[ch].push(*sample); }
        }
    } else {
        // Mono -> réutiliser le vecteur existant
        if let Some(first) = per_channel.get_mut(0) {
            *first = interleaved;
        }
    }

    // Rééchantillonnage individuel si nécessaire
    if source_sr != target_sr {
        for ch in per_channel.iter_mut() {
            *ch = resample(ch, source_sr, target_sr)?;
        }
    }

    Ok(per_channel)
}

// Ancienne fonction conservée pour compatibilité potentielle
#[allow(dead_code)]
pub fn decode_to_mono(path: &PathBuf, target_sr: u32) -> Result<Vec<f32>> {
    let channels = decode_to_channels(path, target_sr)?;
    if channels.is_empty() { return Ok(vec![]); }
    if channels.len() == 1 { return Ok(channels[0].clone()); }
    let len = channels.iter().map(|c| c.len()).min().unwrap_or(0);
    let mut mono = vec![0.0f32; len];
    for ch in &channels { for (i, v) in ch.iter().take(len).enumerate() { mono[i] += *v; } }
    let inv = 1.0 / channels.len() as f32;
    for v in &mut mono { *v *= inv; }
    Ok(mono)
}

/// Rééchantillonne un unique canal audio à la fréquence cible
fn resample(samples: &[f32], source_sr: u32, target_sr: u32) -> Result<Vec<f32>> {
    let mut resampler = FftFixedIn::<f32>::new(
        source_sr as usize,
        target_sr as usize,
        1024,
        1,
        1,
    ).context("Erreur lors de l'initialisation du resampler")?;

    let mut output = vec![vec![0.0f32; resampler.output_frames_max()]; 1];
    let mut resampled = Vec::new();
    for chunk in samples.chunks(1024) {
        let input = vec![chunk.to_vec()];
        let (_, out_len) = resampler.process_into_buffer(&input, &mut output, None)
            .context("Erreur lors du rééchantillonnage")?;
        resampled.extend_from_slice(&output[0][..out_len]);
    }
    Ok(resampled)
}