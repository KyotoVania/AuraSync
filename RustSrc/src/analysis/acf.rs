use num_complex::Complex;
use rustfft::FftPlanner;

/// Analyseur d'autocorrélation utilisant le théorème de Wiener-Khinchin
pub struct AcfAnalyzer {
    pub window_size: usize,
}

impl AcfAnalyzer {
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }

    /// Calcule l'autocorrélation à court terme via FFT (Wiener-Khinchin)
    pub fn analyze(&self, odf_buffer: &[f32]) -> Vec<f32> {
        // Prendre les dernières valeurs selon la taille de fenêtre
        let start = odf_buffer.len().saturating_sub(self.window_size);
        let window = &odf_buffer[start..];

        // Zero-padding pour obtenir une corrélation linéaire correcte
        let fft_len = (window.len() * 2).next_power_of_two();

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_len);
        let ifft = planner.plan_fft_inverse(fft_len);

        // Préparation du buffer avec zero-padding
        let mut buffer: Vec<Complex<f32>> = window
            .iter()
            .map(|&s| Complex::new(s, 0.0))
            .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
            .take(fft_len)
            .collect();

        // 1. Forward FFT
        fft.process(&mut buffer);

        // 2. Power spectrum (|X(f)|²)
        for c in buffer.iter_mut() {
            let power = c.re * c.re + c.im * c.im;
            *c = Complex::new(power, 0.0);
        }

        // 3. Inverse FFT
        ifft.process(&mut buffer);

        // 4. Normalisation et extraction de la partie réelle
        let scale = 1.0 / fft_len as f32;
        buffer.iter()
            .map(|c| c.re * scale)
            .take(window.len())  // Ne garder que la partie utile
            .collect()
    }
}