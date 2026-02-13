import json
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as PathEffects
import argparse
import os
import re

def plot_analysis_visualization(json_path, audio_path, output_image_path):
    """
    Génère une visualisation des résultats d'analyse musicale à partir d'un
    fichier JSON et d'un fichier audio.

    Args:
        json_path (str): Chemin vers le fichier d'analyse JSON
        audio_path (str): Chemin vers le fichier audio
        output_image_path (str): Chemin de sortie pour l'image PNG
    """

    # Charger et parser le fichier JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)

    # Extraire les informations importantes
    audio_duration = analysis_data['audio']['duration']
    cues = [cue for cue in analysis_data['cues'] if 'duration' in cue]

    # Charger le fichier audio et calculer l'enveloppe RMS
    y, sr = librosa.load(audio_path)

    # Calculer l'énergie RMS avec une fenêtre de 512 échantillons
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Créer l'axe temporel pour l'RMS
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Créer une palette de couleurs professionnelle pour les différents types de segments
    def get_base_label(label):
        """Extrait le label de base en supprimant les numéros (ex: chorus_1 -> chorus)"""
        return re.sub(r'_\d+$', '', label)

    base_labels = list(set(get_base_label(cue['type']) for cue in cues))
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(base_labels)))
    color_map = {label: color for label, color in zip(base_labels, colors)}

    # Créer la figure avec les dimensions appropriées
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
                                   gridspec_kw={'height_ratios': [7, 3], 'hspace': 0.1},
                                   sharex=True)

    # Graphique supérieur : Forme d'onde RMS avec remplissage professionnel
    ax1.fill_between(times, 0, rms, color='gray', alpha=0.6, label='RMS Energy')
    ax1.plot(times, rms, 'k-', linewidth=0.5)  # Contour fin pour définition
    ax1.set_ylabel('RMS Energy', fontweight='bold')
    ax1.set_xlim(0, audio_duration)
    ax1.grid(axis='y', linestyle=':', alpha=0.5)
    ax1.tick_params(axis='x', labelbottom=False)  # Masquer les labels X du haut

    # Supprimer les bordures superflues pour un look épuré
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Graphique inférieur : Barre de segments structurels
    for cue in cues:
        start_time = cue['t']
        duration = cue['duration']
        segment_type = cue['type']
        base_type = get_base_label(segment_type)

        # Dessiner le rectangle coloré qui remplit toute la hauteur
        color = color_map[base_type]
        rect = Rectangle((start_time, 0), duration, 1,
                        facecolor=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        ax2.add_patch(rect)

        # Ajouter le label au centre du segment avec effet de contour professionnel
        center_time = start_time + duration / 2
        text = ax2.text(center_time, 0.5, segment_type,
                       ha='center', va='center',
                       fontweight='bold', color='black', fontsize=9)

        # Effet de contour blanc pour lisibilité parfaite (technique professionnelle)
        text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

    # Configuration de l'axe des segments
    ax2.set_xlim(0, audio_duration)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Segments', fontweight='bold')
    ax2.set_yticks([])

    # Supprimer les bordures superflues pour un look épuré
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # Configuration de l'axe X (temps au format mm:ss)
    def format_time(x, pos):
        """Formate le temps en minutes:secondes"""
        minutes = int(x // 60)
        seconds = int(x % 60)
        return f"{minutes}:{seconds:02d}"

    from matplotlib.ticker import FuncFormatter
    ax2.xaxis.set_major_formatter(FuncFormatter(format_time))
    ax2.set_xlabel('Time (min:sec)', fontweight='bold')

    # Titre du graphique
    audio_filename = os.path.basename(audio_path)
    fig.suptitle(f'Music Analysis: {audio_filename}', fontsize=14, fontweight='bold')

    # Ajuster la mise en page et sauvegarder
    plt.subplots_adjust(hspace=0.1)

    # Sauvegarder avec une haute résolution
    plt.savefig(output_image_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Visualisation sauvegardée dans : {output_image_path}")

# --- Point d'entrée du script ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Génère une visualisation des résultats d\'analyse musicale'
    )
    parser.add_argument('--audio', '-a', required=True,
                       help='Chemin vers le fichier audio')
    parser.add_argument('--json', '-j', default='analysis.json',
                       help='Chemin vers le fichier d\'analyse JSON (défaut: analysis.json)')
    parser.add_argument('--output', '-o', default='analysis_visualization.png',
                       help='Chemin de sortie pour l\'image (défaut: analysis_visualization.png)')

    args = parser.parse_args()

    # Vérifier que les fichiers d'entrée existent
    if not os.path.exists(args.json):
        print(f"Erreur: Le fichier JSON '{args.json}' n'existe pas.")
        exit(1)

    if not os.path.exists(args.audio):
        print(f"Erreur: Le fichier audio '{args.audio}' n'existe pas.")
        exit(1)

    # Générer la visualisation
    plot_analysis_visualization(
        json_path=args.json,
        audio_path=args.audio,
        output_image_path=args.output
    )