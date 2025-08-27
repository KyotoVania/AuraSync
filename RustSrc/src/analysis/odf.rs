use num_complex::Complex;
use rustfft::FftPlanner;

/// Paramètres pour la génération de l'ODF
pub struct OdfGenerator {
    pub frame_size: usize,
    pub hop_size: usize,
}

impl OdfGenerator {
    pub fn new(frame_size: usize, hop_size: usize) -> Self {
        Self { frame_size, hop_size }
    }

    /// Génère l'ODF en utilisant le flux spectral agrégé par la médiane
    pub fn generate(&self, samples: &[f32]) -> Vec<f32> {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.frame_size);

        // Création de la fenêtre de Hann manuellement
        let window: Vec<f32> = (0..self.frame_size)
            .map(|i| {
                let t = i as f32 / (self.frame_size - 1) as f32;
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * t).cos())
            })
            .collect();

        let mut odf = Vec::new();
        let mut prev_magnitudes = vec![0.0; self.frame_size / 2 + 1];

        // Calcul du nombre de frames
        let num_frames = (samples.len() - self.frame_size) / self.hop_size + 1;

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_size;
            let end = start + self.frame_size;

            if end > samples.len() {
                break;
            }

            // Application de la fenêtre et préparation pour FFT
            let mut buffer: Vec<Complex<f32>> = samples[start..end]
                .iter()
                .zip(window.iter())
                .map(|(s, w)| Complex::new(s * w, 0.0))
                .collect();

            // FFT
            fft.process(&mut buffer);

            // Calcul des magnitudes (seulement la moitié positive du spectre)
            let magnitudes: Vec<f32> = buffer[..self.frame_size / 2 + 1]
                .iter()
                .map(|c| (c.re * c.re + c.im * c.im).sqrt())
                .collect();

            // Calcul du flux spectral avec agrégation par médiane
            let mut spectral_diffs: Vec<f32> = magnitudes
                .iter()
                .zip(prev_magnitudes.iter())
                .map(|(current, prev)| {
                    // Passage en échelle logarithmique pour plus de robustesse
                    let log_current = (current + 1e-10_f32).ln();
                    let log_prev = (prev + 1e-10_f32).ln();
                    (log_current - log_prev).max(0.0)  // Redressement demi-onde
                })
                .collect();

            // Calcul de la médiane
            if !spectral_diffs.is_empty() {
                spectral_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median_idx = spectral_diffs.len() / 2;
                odf.push(spectral_diffs[median_idx]);
            } else {
                odf.push(0.0);
            }

            prev_magnitudes = magnitudes;
        }

        odf
    }
}