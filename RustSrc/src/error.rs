use thiserror::Error;

#[derive(Error, Debug)]
pub enum AnalysisError {
    #[error("Erreur de décodage du fichier : {0}")]
    Decode(String),
    #[error("BPM non trouvé")]
    BpmNotFound,
    #[error("Erreur d'entrée/sortie : {0}")]
    Io(#[from] std::io::Error),
}