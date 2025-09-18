# Audit Comparatif: RealBPMModule v2.0.0-beattracking vs Mixxx (QM-DSP / SoundTouch)

Auteur: Agent Junnie (Audio Visual Engine)
Date: 2025-09-18

Objectif: Ingénierie inverse et comparaison du pipeline BPM/Beat Tracking de Mixxx avec notre RealBPMModule v2.0.0, pour identifier des principes de robustesse et des techniques à intégrer.

---

1) Cartographie des Pipelines

A. Notre pipeline (RealBPMModule v2.0.0)
- ODF: Complex Spectral Difference (mono-bande) + lissage léger
  • Référence code: src/modules/RealBPMModule.cpp → extractComplexSpectralDifferenceODF(...)
- Détection des candidats: pics locaux avec seuil adaptatif (mean + 1.0·std) + NMS temporelle (fenêtre min correspondant à 300 BPM)
  • Référence code: detectBeatCandidates(...)
- Estimation du tempo: Tempogramme par autocorrélation locale (log-spaced 120 bins) + renfort harmoniques (×0.5, ×2)
  • Référence code: computeTempogram(...)
- Tracking: Programmation dynamique (Viterbi-like) sur candidats, coût de transition = support tempogramme + bonus harmonique + préférence musicale (réduite) − pénalité de changement de tempo
  • Référence code: trackBeatsWithDynamicProgramming(...), computeTransitionCost(...)
- Sortie/Confiance: BPM par médiane d’intervalles; confiance = 1 − 2·(écart-type/intervalle médian)
  • Référence code: generateBeatTrackingResult(...)

B. Pipeline Mixxx (QM-DSP par défaut, SoundTouch en alternative)
- Orchestrateur: AnalyzerBeats sélectionne un plugin (par défaut Queen Mary), gère préférences (tempo fixe, ré-analyse, fast analysis)
  • mixxx/src/analyzer/analyzerbeats.{h,cpp}
- ODF: DF_COMPLEXSD (Complex Spectral Difference) via QM-DSP DetectionFunction
  • mixxx/src/analyzer/plugins/analyzerqueenmarybeats.cpp: makeDetectionFunctionConfig → DFType = DF_COMPLEXSD, dbRise=3
- Paramètres temporels: step ≈ 11.61 ms (kStepSecs), window = nextPowerOfTwo(fs / 50 Hz) → ~1024 @ 44.1 kHz
  • idem
- Estimation tempo/beat: TempoTrackV2 (QM-DSP) calcule beatPeriod puis les frappes (calculateBeats)
  • mixxx/src/analyzer/plugins/analyzerqueenmarybeats.cpp (TempoTrackV2)
- Beat grid: liste de positions d’impulsions (FramePos) post-traitées; pas de programmation dynamique explicite côté Mixxx (délégué à QM-DSP)
- Confiance: aucune métrique de confiance explicite exposée par AnalyzerQueenMaryBeats/AnalyzerBeats; stockage via Beats/BeatFactory avec versioning, pas de score de fiabilité direct
  • Recherche «confidence» dans mixxx: aucune occurrence pertinente dans l’analyse BPM/beat
- Alternative: AnalyzerSoundTouchBeats (soundtouch::BPMDetect) renvoie un BPM global (sans grille détaillée)
  • mixxx/src/analyzer/plugins/analyzersoundtouchbeats.cpp

Conclusion carto:
- Similarités fortes sur l’ODF (Complex SD) et la logique d’analyse temporelle locale
- Mixxx délègue l’estimation de tempo/grille au module robuste TempoTrackV2 (QM-DSP) plutôt qu’à une DP custom
- Mixxx met l’accent sur l’ingénierie produit (plugins, préférences, fast analysis, ré-analyse conditionnelle, versioning)

---

2) Analyse Comparative par Composant

2.1 Détection des Attaques (ODF)
- Mixxx: Complex Spectral Difference via QM-DSP avec paramètres éprouvés (dbRise=3, pas d’adaptive whitening)
- Nous: Complex SD maison monobande, fenêtre Hann, lissage léger
- Écart clé: Mixxx utilise step/window calibrés (~11.6 ms, ~1024) et un ODF standardisé QM-DSP; nous pourrions harmoniser nos paramètres de fenêtre/pas ou exposer un mode «QM-like».

2.2 Estimation de Tempo
- Mixxx: TempoTrackV2 (algorithme robuste incluant suivi de tempo et harmonics handling intégré) sur la série ODF; pas d’autocorrélation explicite côté Mixxx code (caché dans QM-DSP)
- Nous: Tempogramme par autocorrélation locale + bonus harmoniques simples
- Écart clé: La logique d’évidence de tempo de QM-DSP paraît plus intégrée (combinaison de méthodes + heuristiques). Nos renforts harmoniques sont plus simples.

2.3 Génération de Grille
- Mixxx (QM): calcul direct des battements (calculateBeats) après beatPeriod; pas de DP explicite
- Nous: DP sur une liste de candidats avec coûts modulaires (support tempogramme, préférences, pénalités)
- Avantage actuel: Notre DP est explicable et modulable (greffons de coûts). Avantage Mixxx: robustesse éprouvée et paramètres calibrés (step/window/tempo tracker).

2.4 Calcul de la Confiance
- Mixxx: pas de métrique de confiance exposée par AnalyzerBeats; la fiabilité est implicite (grille stable) et encadrée par politiques produit (réanalyse, verrouillage BPM, versioning)
- Nous: confiance basée sur la cohérence des intervalles (écart-type relatif)
- Opportunité: enrichir notre score de confiance avec des signaux «santé» inspirés par Mixxx (stabilité locale du tempo, couverture de grille, persistance du pic tempo, consensus mono/stéréo, etc.)

---

3) Divergences Conceptuelles Clés (explicatives de robustesse)
- Paramétrage temporel calibré (Mixxx): step ~11.6 ms, window ~20 ms @ 44.1 kHz, cohérent avec QM-DSP et pratique DJ (≈ 86 Hz)
- Piste principale unique et éprouvée: délégation à TempoTrackV2 (intégrant la gestion d’octaves et transitions de tempo)
- Architecture plugin + politiques de réanalyse: évite les faux positifs persistants, assure la fraîcheur des résultats en fonction des préférences et versions
- Absence d’un score de «confiance» isolé: la robustesse vient de la stabilité du tracker et du contrôle de flux (reanalyze/lock) plutôt que d’un score ex post

---

4) Techniques/Concepts Mixxx prometteurs à greffer (3–5)

T1. Paramètres Step/Window «QM-like» pour l’ODF
- Détails: step ≈ 0.01161 s; window = nextPow2(fs / 50 Hz)
- Attendu: meilleure stabilité des pics ODF, résolution temporelle régulière; confiance accrue par intervalles plus cohérents
- Réf: analyzerqueenmarybeats.cpp (kStepSecs, kMaximumBinSizeHz)

T2. Post-traitement «Octave Sanity Check» sur le BPM final
- Détails: réévaluer 0.5×/1×/2× du BPM final en regardant l’évidence tempogramme/ODF et la consistance de grille
- Attendu: réduction des rares erreurs 70↔140, 80↔160; robustesse
- Réf: logique implicite de TempoTrackV2 + pratiques de l’industrie

T3. Mode «Fixed Tempo Assumption» et «Fast Analysis» inspirés Mixxx
- Détails: exposer un paramètre «tempo fixe» (durée courte) et un mode d’analyse rapide (N premières secondes)
- Attendu: robustesse perçue (moins de faux changements), rapidité; configurable sans impacter exactitude par défaut
- Réf: AnalyzerBeats::initialize (m_bPreferencesFixedTempo, m_bPreferencesFastAnalysis)

T4. Métrique de Confiance multi-facteurs (stabilité/coverage/persistance)
- Détails: combiner notre std/median avec: persistance du pic tempo, ratio coverage (grille vs durée), variance locale du BPM, consensus canaux
- Attendu: score de confiance plus représentatif et utile pour l’UX
- Réf: nos generateBeatTrackingResult + indices issus de Mixxx (absence de score → place pour mieux)

T5. Filtres harmoniques étagés à la QM (comb-like sur ODF) en plus de l’ACF
- Détails: comb filters multi-harmoniques (1×, 2×, 3×, 4×) pondérés 1/h^2
- Attendu: salience de tempo plus nette sur textures denses
- Réf: Notre v4 hybride (BPMModuleAlt.cpp) et littérature; cohérent avec l’esprit de TempoTrackV2

---

5) Réponses aux Questions Ciblées
- ODF: Mixxx utilise Complex Spectral Difference (comme nous), via QM-DSP; pas multi-bandes par défaut
- Estimation de tempo: pilotée par TempoTrackV2 (combinaison interne de méthodes), gère l’ambiguïté d’octave au cœur de l’algorithme
- Grille de pulsations: pas de DP explicite; calcul direct des beats à partir de beatPeriod suivi
- Confiance: pas de métrique explicite; robustesse par design + politiques produit

---

6) Synthèse
Mixxx et notre v2 partagent une base ODF similaire. Mixxx gagne en robustesse grâce à des choix paramétriques et à l’usage d’un tracker de tempo intégré (TempoTrackV2) avec des pratiques d’ingénierie (plugins, reanalyse, fast mode). Les pistes T1–T5 ci-dessus constituent des «greffons» raisonnables et peu risqués pour améliorer la robustesse et la confiance de notre module sans compromettre l’exactitude atteinte (100% sur notre corpus actuel).
