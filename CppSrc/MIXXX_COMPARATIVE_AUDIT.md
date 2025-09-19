# Directive Stratégique – Audit d'Intégration et Analyse Comparative Face à Mixxx/QM-DSP

Auteur: Agent Junie (Audio Visual Engine)
Date: 2025-09-19

Objet: Analyse comparative exhaustive entre notre RealBPMModule v2.0.0-beattracking et l'implémentation Mixxx basée sur QM-DSP, en vue de décider d'une intégration totale, hybride, ou d'une amélioration native.

---

1) Introduction – Redémarrage stratégique axé performance
Nous privilégions désormais une approche pragmatique orientée résultats. Objectif immédiat: atteindre la précision et la robustesse de Mixxx. Cet audit compare point par point notre pipeline et celui de Mixxx (QM-DSP) et conclut par une recommandation opérationnelle.

---

2) Méthodologie et sources
- Notre code: src/modules/RealBPMModule.cpp (extractComplexSpectralDifferenceODF, detectBeatCandidates, computeTempogram, trackBeatsWithDynamicProgramming, postProcessOctaveCorrection, generateBeatTrackingResult)
- Mixxx: mixxx/src/analyzer/plugins/analyzerqueenmarybeats.{h,cpp}; mixxx/src/analyzer/analyzerbeats.cpp
  • Paramètres clés observés: DF_COMPLEXSD, dbRise=3, step kStepSecs=0.01161 s, window = nextPowerOfTwo(fs / 50 Hz)
  • Tracking: TempoTrackV2 (qm-dsp/dsp/tempotracking/TempoTrackV2.h)
- Littérature QM-DSP (synthèse): combinaisons autocorrélation + filtres en peigne + heuristiques de suivi du tempo (Davies & Plumbley; Dixon; Ellis; Brossier)

---

3) Analyse exhaustive par étape du pipeline

Étape 1 – Fonction de Détection d'Attaques (ODF)
- Confirmation ODF: Mixxx/QM-DSP et notre module utilisent tous deux la Complex Spectral Difference (CSD / DF_COMPLEXSD). Dans Mixxx: makeDetectionFunctionConfig → DFType=DF_COMPLEXSD, dbRise=3, whitening désactivé. Côté AVE: CSD mono-bande avec fenêtre Hann et lissage léger.
- Fenêtrage et pas temporel: Mixxx fixe un pas temporel constant kStepSecs≈11.61 ms et choisit une fenêtre window = nextPow2(fs/50 Hz):
  • fs=44.1 kHz → step≈512 échantillons; window≈1024 (fs/50=882 → pow2=1024)
  • fs=48 kHz → step≈557; window≈1024 (960→1024)
  • fs=96 kHz → step≈1115; window≈2048 (1920→2048)
- Pourquoi une résolution temporelle constante est préférable à un pas en échantillons fixe:
  • Invariance au taux d'échantillonnage: une granularité fixe en temps maintient une résolution (≈86 Hz) identique entre 44.1/48/96 kHz. Un pas fixe en samples induit une résolution dépendante du fs et des biais de tempo.
  • Stabilité de phase/peigne: les filtres en peigne et l'autocorrélation opèrent mieux quand le pas temporel est constant; cela réduit le jitter des pics ODF et améliore l'alignement du beat period.
  • Cohérence des métriques: lissage, seuils adaptatifs et tempogrammes se calibrent en secondes, pas en samples; cela facilite la généralisation cross-device.
- Impact attendu sur la robustesse: + cohérence des pics ODF, + qualité d'estimation du beat period, – sensibilité au bruit de sr. C'est un facteur clé de la robustesse de Mixxx.

Étape 2 – Détection des pulsations candidates
- «Boîte noire» QM-DSP: TempoTrackV2 consomme la série ODF continue (sans binarisation forte). L'algorithme déduit implicitement les candidats via l'évidence périodique et le suivi, plutôt qu'un seuillage agressif. Par comparaison, notre seuil mean + 1·std + NMS est plus «tranchant» (risque de rater des battements faibles).
- Évaluation d'agressivité: Mixxx/QM-DSP est moins agressif en amont (préserve l'information), plus sélectif en aval via le suivi; notre approche est plus agressive au pré-traitement (seuillage + NMS), puis flexible via DP.
- Notre avantage à conserver: la non-maximum suppression explicite + test de maximum local produisent un ensemble de candidats propre, contrôlable et traçable. Dans une architecture hybride, nous pouvons laisser QM-DSP fournir un beat period/beat phase robuste tout en conservant NMS/local-max en sur-couche pour la régularisation et l'explicabilité.

Étape 3 – Estimation du tempo et périodicité
- Coeur de la divergence: TempoTrackV2. Selon la littérature et l'implémentation observée, il combine: (a) autocorrélation/FFT pour la périodicité, (b) filtres en peigne multi-harmoniques pondérés, (c) heuristiques de continuité et de phase, (d) suivi du beat period avec contraintes (ex. inertie, bornes BPM), puis (e) génération de frappes (calculateBeats).
- Supériorité structurelle vs notre tempogramme ACF seul:
  • Fusion d'évidences: comb + ACF réduit les confusions 0.5×/2×/3× mieux que des pondérations manuelles.
  • Suivi temporel intégré: le beat period est suivi au fil du temps (pas seulement choisi globalement), ce qui gère les transitions de tempo et stabilise la phase.
  • Moins de réglages fragiles: les heuristiques intégrées encapsulent des décennies de calibrations (dbRise, hop constant, bornes BPM), difficiles à recréer vite.
- Gestion des harmoniques (octaves): Nos pondérations manuelles (0.5×/2×) restent naïves; TempoTrackV2 opère une gestion intégrée des harmoniques via le peigne et les heuristiques de continuité, moins sujette aux sauts 70↔140/80↔160.

Étape 4 – Suivi de pulsation et post-traitement
- Mixxx/QM-DSP: calculateBeats fournit directement la grille (pas de DP exposée côté Mixxx). Transparence limitée sur la fonction de coût, mais robustesse éprouvée.
- Notre architecture modulaire: DP (coûts explicites: support tempogramme, bonus harmoniques, pénalités de variation), correction d'octave a posteriori, métriques de santé (cohérence d'intervalles, couverture). Avantage: explicabilité, configurabilité, et facilité d'AB testing. C'est notre atout principal à préserver.

---

4) Tableaux comparatifs (validés)

A. Par étape du pipeline
| Étape | Mixxx / QM-DSP | AVE RealBPMModule |
| - | - | - |
| ODF | Complex SD (DF_COMPLEXSD), dbRise=3, hop=0.01161 s, window=nextPow2(fs/50), whitening=off | Complex SD mono-bande, Hann, hop configurable (actuel en samples), lissage léger |
| Candidats | Pas de seuillage dur; evidence périodique + suivi | Pics locaux + seuil mean+1σ + NMS temporelle |
| Tempo | TempoTrackV2: ACF+peignes+heuristiques; suivi du beat period | Tempogramme ACF, pondérations 0.5×/2× |
| Grille | calculateBeats (QM) → positions | DP avec coûts explicites |
| Confiance | Non exposée | Écart-type relatif + métriques santé |

B. Paramètres temporels
| Paramètre | Mixxx | AVE actuel | Impact |
| - | - | - | - |
| Pas (hop) | 0.01161 s (≈86 Hz) | en samples (varie en s selon fs) | + robustesse cross-fs côté Mixxx |
| Fenêtre | nextPow2(fs/50) | fixe ou liée à N interne | + salience CSD et stabilité des bins côté Mixxx |

---

5) Synthèse des facteurs de performance clés (Top 3)
1) Pas temporel constant + fenêtre liée à fs (nextPow2(fs/50)) qui stabilisent l'ODF et la périodicité.
2) TempoTrackV2: fusion ACF+peignes + suivi de tempo/phase → meilleure désambiguïsation d'octave et transitions.
3) Paramétrage éprouvé et flux produit (plugins, fast analysis, réanalyse conditionnelle) → moins d'états incohérents.

---

6) Recommandation stratégique
Option B – Intégration Hybride (Recommandée)
- Principe: utiliser QM-DSP pour l'ODF et le tracking (TempoTrackV2) afin d'obtenir la robustesse/rapidité; conserver notre «couche supérieure» (DP optionnelle pour régulariser, correction d'octave avancée, métriques de santé, configurabilité produit).
- Pourquoi pas A (intégration totale)? Nous perdrions notre transparence/coût explicite et nos métriques santé, différenciateurs utiles pour diagnostics et UX.
- Pourquoi pas C (amélioration native seule)? Faisable mais plus long/risqué pour atteindre la parité; TempoTrackV2 encapsule des heuristiques non triviales.

Plan d'exécution (court terme 1–2 sprints):
- Intégrer QM-DSP côté ODF+Tempo (comme Mixxx): hop=0.01161 s; window=nextPow2(fs/50); DF_COMPLEXSD (dbRise=3).
- Conserver/brancher notre DP en option «regularize grid» au-dessus des beats QM (si nécessaire pour cas limites), sans bloquer le chemin QM pur.
- Maintenir nos métriques santé et l'octave sanity-check en post-process sur la grille QM.
- Exposer des modes «fast analysis» et «tempo fixe» inspirés Mixxx.

Risques & mitigations:
- Divergences de dépendances/ABI: utiliser la même version de QM-DSP que Mixxx; valider sous nos OS cibles.
- Lissage des frontières d'analyse: aligner notre frameRate interne avec kStepSecs.

---

7) Réponses ciblées (validation du rapport initial)
- ODF: Oui, Complex Spectral Difference des deux côtés. Paramètres QM confirmés: dbRise=3; whitening off.
- Candidats: QM-DSP est moins agressif en pré-traitement; notre NMS/local-max est un atout à conserver en sur-couche.
- Tempo/périodicité: TempoTrackV2 est structurellement supérieur à un tempogramme ACF seul (comb + heuristiques + suivi continu).
- Harmoniques: Notre pondération 0.5×/2× est naïve vs gestion intégrée du peigne TempoTrackV2.
- Suivi/post-traitement: Notre architecture DP/correction/métriques est plus transparente et reste un différenciateur clé.

---

8) Conclusion
Pour converger rapidement vers un niveau industriel, adoptons une intégration hybride: QM-DSP pour l'ODF et le tracking (robustesse/rapidité), tout en capitalisant sur notre couche supérieure (explicabilité, métriques, configurabilité). Cette stratégie minimise le risque, maximise le gain de performance, et préserve notre avantage produit.
