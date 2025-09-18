# Feuille de Route v2.1.0 – Robustesse & Confiance (inspirée de Mixxx)

Auteur: Agent Junnie
Date: 2025-09-18
Objectif: Intégrer des «bonnes pratiques» issues de Mixxx pour augmenter la robustesse et la confiance de RealBPMModule tout en préservant l’exactitude (actuellement 100% sur notre corpus).

---

Priorisation (faible risque de régression → fort impact sur confiance/robustesse): C1, R1, R2, R3, R4

Glossaire:
- Cx = Hypothèse Confiance
- Rx = Hypothèse Robustesse

---

C1. Confiance multi‑facteurs (Mixxx‑inspired health checks)
- Hypothèse: En combinant l’écart‑type relatif des intervalles (actuel) avec des facteurs de stabilité, on obtient un score de confiance plus représentatif.
- Détails techniques:
  - Persistance du pic tempo local: fraction d’images tempogramme où le bin dominant reste dans ±8%.
  - Coverage de grille: ratio (#beats estimés / durée/intervalle_médian) dans [0.85, 1.15].
  - Stabilité BPM locale: écart‑type des intervalles dans des fenêtres glissantes (2–4 s).
  - Consensus canaux (si stéréo): différence de BPM estimé par canal mono mixdowns A/B.
- Sortie: confidence = clamp( w1·cohérence_intervalles + w2·persistance + w3·coverage + w4·stabilité_locale − w5·désaccord_stéréo ).
- Attendu: Confiance plus corrélée à la fiabilité perceptive; utile pour UX et décisions aval.
- Impact risque: faible (métrique purement dérivée). Priorité: Haute.

R1. Post‑traitement correction d’octave (a posteriori)
- Hypothèse: Un check 0.5×/1×/2× basé sur l’évidence tempogramme et la consistance de grille réduit les rares erreurs d’octave.
- Détails techniques:
  - Calculer score_total(bpm_k) = α·évidence_tempogramme(bpm_k) + β·coverage_grille(bpm_k) + γ·stabilité_intervalles(bpm_k), pour k ∈ {0.5×, 1×, 2×} du BPM courant.
  - Sélectionner bpm_k* max si gain > seuil (p.ex. 5%).
- Attendu: Robustesse accrue sur tempos multiples (70/140, 80/160).
- Impact risque: faible (post‑filtrage). Priorité: Haute.

R2. Paramètres ODF «QM‑like» (step/window)
- Hypothèse: Harmoniser step ≈ 11.61 ms et window ≈ nextPow2(fs/50 Hz) stabilise l’ODF.
- Détails techniques:
  - Exposer paramètres dans config (frameSize/hopSize option «qmLike=true» qui fixe hop = round(sr*0.01161), frame = nextPow2(sr/50)).
- Attendu: Meilleure définition des pics ODF et cohérence des intervalles → confiance accrue.
- Impact risque: moyen (peut décaler candidats). Activer derrière un flag. Priorité: Moyenne‑haute.

R3. Mode «Fixed Tempo Assumption» et «Fast Analysis»
- Hypothèse: À l’instar de Mixxx, proposer: (a) tempo fixe (pénaliser changements rapides), (b) fast analysis (analyser N premières secondes) améliore robustesse et UX.
- Détails techniques:
  - Paramètres: fixedTempo=true/false (renforce pénalités de changement), fastAnalysisSeconds ∈ [10, 45].
  - Implémentation: appliquer poids supplémentaires dans computeTransitionCost et tronquer l’ODF si fastAnalysisSeconds>0.
- Attendu: Moins de faux «tempo shifts», gain de temps d’analyse.
- Impact risque: faible (opt‑in). Priorité: Moyenne.

R4. Tempogramme hybride avec peigne harmonique (comb‑like)
- Hypothèse: Ajouter une mesure par filtres peigne (pondérés 1/h^2) en complément de l’ACF renforce l’évidence du tempo sur textures denses.
- Détails techniques:
  - Implémenter une deuxième carte d’évidence (comb) et la fondre: evidence = (1−λ)·ACF + λ·COMB (λ≈0.3).
- Attendu: Salience de tempo plus nette, surtout pour percussions polyrythmiques.
- Impact risque: moyen. Activer derrière flag. Priorité: Moyenne.

C2. Exposition de «health flags» dans la sortie
- Hypothèse: Signaux binaires (p.ex. «tempo drift suspected», «octave ambiguity high», «low coverage») aident le débogage et la UI.
- Détails techniques: calculés avec les mêmes métriques que C1/R1.
- Impact risque: faible. Priorité: Moyenne.

---

Plan d’Intégration & Validation Continue
1) Implémenter C1 (confiance multi‑facteurs) derrière un flag; compiler, tests unitaires; baseline_test.ps1 sur corpus complet; documenter. Cible: confiance moyenne ↑ sans baisse d’exactitude.
2) Implémenter R1 (octave check) en post‑traitement; re‑tests; vérifier absence de régressions; documenter.
3) Implémenter R2 (mode QM‑like step/window) caché derrière config; A/B tests vs défaut; tuner paramètres; documenter.
4) Implémenter R3 (fixed tempo, fast analysis) – paramètres config; vérifier gains perçus et temps d’analyse.
5) Implémenter R4 (tempogramme hybride ACF+COMB) sous flag; évaluer gains sur morceaux denses.
6) Implémenter C2 (health flags) dans JSON de sortie; adapter baseline_test.ps1 si besoin pour rapporter ces indicateurs.

Critères de succès v2.1.0
- Exactitude maintenue à 100% sur corpus actuel; pas d’erreur d’octave.
- Confiance moyenne +15–25% vs v2.0.0, distribution plus resserrée.
- Robustesse démontrée sur corpus élargi: 95%+ exactitude.
- Documentation: MIXXX_COMPARATIVE_AUDIT.md (présent) + changelog + paramètres exposés.
