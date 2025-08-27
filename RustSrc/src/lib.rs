pub mod analysis;
pub mod config;
pub mod model;
pub mod error;

pub use config::{MergeMode, AnalysisConfig};
use model::{AnalysisResult, Metadata, TempoAnalysis};
use error::AnalysisError;

pub fn run(config: &AnalysisConfig) -> Result<AnalysisResult, AnalysisError> {
    // 1. Décodage en mono (comme dans la V1)
    let samples = analysis::decoder::decode_to_mono(&config.file_path, config.target_sr)
       .map_err(|e| AnalysisError::Decode(e.to_string()))?;
    
    let num_samples = samples.len();
    let duration_seconds = num_samples as f32 / config.target_sr as f32;

    // 2. Génération de l'ODF (directement sur le signal mono)
    let odf_gen = analysis::odf::OdfGenerator::new(config.frame_size, config.hop_size);
    let odf_signal = odf_gen.generate(&samples);
    let odf_sample_rate = config.target_sr as f32 / config.hop_size as f32;

    // 4. Analyse et Estimation
    let acf_window_size = (3.0 * odf_sample_rate) as usize;
    let acf_analyzer = analysis::acf::AcfAnalyzer::new(acf_window_size);
    let mut bpm_estimator = analysis::estimator::BpmEstimator::new(config.min_bpm, config.max_bpm);
    
    let analysis_hop = (odf_sample_rate as usize) / 2;
    let mut final_bpm = None;

    for window_start in (0..odf_signal.len()).step_by(analysis_hop) {
        let window_end = (window_start + acf_window_size).min(odf_signal.len());
        if window_end - window_start < acf_window_size / 2 { break; }
        let acf = acf_analyzer.analyze(&odf_signal[..window_end]);
        if let Some(bpm) = bpm_estimator.estimate(&acf, odf_sample_rate) {
            final_bpm = Some(bpm);
        }
    }

    let global_bpm = final_bpm.ok_or(AnalysisError::BpmNotFound)?;

    // 5. Construire la structure de résultat
    Ok(AnalysisResult {
        metadata: Metadata {
            file_path: config.file_path.clone(),
            sample_rate: config.target_sr,
            channels: 1, // Toujours 1 car on convertit en mono
            duration_seconds,
        },
        tempo_analysis: TempoAnalysis {
            global_bpm,
        },
    })
}