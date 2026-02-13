# Music Analysis Visualizer

Ce script Python génère des visualisations statiques à partir de fichiers d'analyse musicale JSON et de leurs fichiers audio correspondants.

## Fonctionnalités

- **Courbe d'énergie RMS** : Affichage de l'enveloppe d'énergie du signal audio
- **Segments structurels** : Visualisation colorée des différentes sections musicales
- **Formatage temporel** : Axe X au format minutes:secondes (mm:ss)
- **Couleurs cohérentes** : Groupement automatique des segments similaires (ex: tous les "chorus" ont la même couleur)

## Prérequis

```bash
pip install librosa matplotlib numpy
```

## Utilisation

### Utilisation basique
```bash
python music_analysis_visualizer.py --audio path/to/your/song.mp3
```

### Utilisation complète
```bash
python music_analysis_visualizer.py --audio song.mp3 --json analysis.json --output visualization.png
```

### Arguments

- `--audio` / `-a` : **Obligatoire**. Chemin vers le fichier audio
- `--json` / `-j` : Chemin vers le fichier JSON d'analyse (défaut: `analysis.json`)
- `--output` / `-o` : Nom du fichier de sortie PNG (défaut: `analysis_visualization.png`)

## Format du fichier JSON

Le script attend un fichier JSON avec cette structure :

```json
{
  "audio": {
    "duration": 236.13333333333333
  },
  "cues": [
    {
      "t": 0.0,
      "duration": 6.135872840881348,
      "type": "intro"
    },
    {
      "t": 6.135872840881348,
      "duration": 7.999274253845215,
      "type": "drop"
    }
  ]
}
```

## Sortie

Le script génère une image PNG (1200x600 pixels par défaut) avec :

- **Partie supérieure (70%)** : Courbe RMS en noir sur fond blanc
- **Partie inférieure (30%)** : Bandeaux colorés représentant les segments musicaux avec leurs labels

## Exemples

Pour analyser un fichier avec les paramètres par défaut :
```bash
python music_analysis_visualizer.py --audio my_song.mp3
```

Cela cherchera `analysis.json` dans le répertoire courant et créera `analysis_visualization.png`.