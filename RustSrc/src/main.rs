use clap::Parser;
use std::path::PathBuf;
use anyhow::Result;

use bpm_detector::{AnalysisConfig, MergeMode, run};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg()]
    file_path: PathBuf,
    #[arg(long, default_value_t = 60.0)]
    min_bpm: f32,
    #[arg(long, default_value_t = 180.0)]
    max_bpm: f32,
    #[arg(long, value_enum, default_value_t = MergeMode::Max)]
    merge_mode: MergeMode,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 1. Créer la configuration à partir des arguments
    let config = AnalysisConfig {
        file_path: args.file_path,
        target_sr: 44100,
        frame_size: 2048,
        hop_size: 512,
        min_bpm: args.min_bpm,
        max_bpm: args.max_bpm,
        merge_mode: args.merge_mode,
    };

    // 2. Appeler la bibliothèque
    match run(&config) {
        Ok(analysis_result) => {
            // 3. Sérialiser le résultat en JSON et l'afficher
            let json_output = serde_json::to_string_pretty(&analysis_result)?;
            println!("{}", json_output);
        }
        Err(e) => {
            eprintln!("Erreur lors de l'analyse : {}", e);
        }
    }

    Ok(())
}