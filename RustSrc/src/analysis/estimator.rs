/// Estimateur de BPM avec stabilisation
pub struct BpmEstimator {
    pub min_bpm: f32,
    pub max_bpm: f32,
    pub history: Vec<f32>,
    pub history_size: usize,
}

impl BpmEstimator {
    pub fn new(min_bpm: f32, max_bpm: f32) -> Self {
        Self {
            min_bpm,
            max_bpm,
            history: Vec::new(),
            history_size: 10,
        }
    }

    /// Estime le BPM à partir du vecteur d'autocorrélation
    pub fn estimate(&mut self, acf: &[f32], odf_sample_rate: f32) -> Option<f32> {
        // Conversion BPM en plage de lags
        let max_period_s = 60.0 / self.min_bpm;
        let min_period_s = 60.0 / self.max_bpm;
        let min_lag = (min_period_s * odf_sample_rate).round() as usize;
        let max_lag = (max_period_s * odf_sample_rate).round() as usize;

        if max_lag >= acf.len() || min_lag >= max_lag {
            return None;
        }

        // Recherche du pic maximum dans la plage de lags
        let search_range = &acf[min_lag..=max_lag.min(acf.len() - 1)];
        let (peak_local_idx, &peak_value) = search_range
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;

        // Vérification de la qualité du pic (seuil minimal)
        if peak_value < 0.1 {
            return None;
        }

        let peak_lag = min_lag + peak_local_idx;

        // Conversion lag -> BPM
        let bpm = (odf_sample_rate * 60.0) / peak_lag as f32;

        // Stabilisation avec médiane mobile
        self.history.push(bpm);
        if self.history.len() > self.history_size {
            self.history.remove(0);
        }

        // Calcul de la médiane
        let mut sorted_history = self.history.clone();
        sorted_history.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_bpm = sorted_history[sorted_history.len() / 2];

        Some(median_bpm)
    }
}