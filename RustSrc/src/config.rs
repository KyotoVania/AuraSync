use clap::ValueEnum;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct AnalysisConfig {
    pub file_path: PathBuf,
    pub target_sr: u32,
    pub frame_size: usize,
    pub hop_size: usize,
    pub min_bpm: f32,
    pub max_bpm: f32,
    pub merge_mode: MergeMode,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum MergeMode {
    Mean,
    Sum,
    Max,
}