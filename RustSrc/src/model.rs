use serde::{Serialize, Deserialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug)]
pub struct AnalysisResult {
    pub metadata: Metadata,
    pub tempo_analysis: TempoAnalysis,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Metadata {
    pub file_path: PathBuf,
    pub sample_rate: u32,
    pub channels: usize,
    pub duration_seconds: f32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TempoAnalysis {
    pub global_bpm: f32,
}