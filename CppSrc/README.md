Audio Visual Engine (AVE) – Analysis Tool

This repository contains a standalone audio analysis tool and core library for tempo/beat tracking, onsets, tonality, structure, and cue synthesis.

The tool ships with two BPM engines:
- QM‑DSP (Mixxx‑identical): uses Queen Mary DSP DetectionFunction + TempoTrackV2 for robust, Mixxx‑like results.
- Native engine: transparent, modular pipeline with explicit cost functions and health metrics.

Quick Start
- Build (Debug profile configured by CLion):
  - Build main executable: cmake --build <your-cmake-build-dir> --target ave_analysis
  - Build tests (optional): cmake --build <your-cmake-build-dir> --target ave_tests
- Run analysis:
  - ave_analysis <input.wav> [output.json]
    • Example: ave_analysis tests\assets\SimonTheDivergentStar116BpmFm.wav analysis.json
- Run tests:
  - <your-cmake-build-dir>\bin\ave_tests.exe

Command‑line usage
- ave_analysis <input.wav> [output.json]
  - input.wav: path to a mono/stereo PCM WAV file
  - output.json (optional): where to save the JSON analysis (defaults to analysis.json in the working dir)

How configuration works
- Configuration is applied per module from the application. In the provided ave_analysis executable, module configuration is set in src\main.cpp using pipeline->setModuleConfig(...).
- If you embed ave_core in your own application, set module configuration programmatically the same way before running the pipeline.
- The sample main.cpp currently sets the BPM module to use the QM‑DSP engine by default.

Configure BPM analysis
- Engine selection (recommended)
  - engine: "qm" → Use QM‑DSP (Mixxx‑identical: DetectionFunction + TempoTrackV2)
  - engine: "native" → Use the native AVE tracker (ODF + tempogram + DP + octave correction)
- Common parameters
  - minBPM (float): minimum tempo allowed (default 60)
  - maxBPM (float): maximum tempo allowed (default 200)
  - frameSize (size_t): STFT window size for native pipeline (default 1024)
  - hopSize (size_t): STFT hop size for native pipeline (default 512)
  - octaveCorrection (bool): apply 0.5x/1x/2x sanity check grid post‑process (default true)
  - fixedTempo (bool): discourage tempo changes over time (default false)
  - fastAnalysisSeconds (double): analyze only the first N seconds (10–45) for speed (default 0: off)
- Advanced (native engine only)
  - qmLike (bool): use QM‑like hop/window (hop≈0.01161 s, window≈nextPow2(fs/50)) while staying in native mode
  - hybridTempogram (bool): blend ACF with comb‑like evidence
  - combLambda (double [0..1]): comb/ACF blend weight when hybridTempogram=true (default 0.3)
  - combHarmonics (int 2..8): number of comb harmonics (default 4)

BPM config examples
- Mixxx‑identical results (default in ave_analysis):
  { "engine": "qm", "minBPM": 20, "maxBPM": 220, "octaveCorrection": true }
- Native engine with QM‑like framing and fixed tempo assumption:
  { "engine": "native", "qmLike": true, "fixedTempo": true, "minBPM": 60, "maxBPM": 200 }
- Fast analysis for quick previews (first 20 seconds):
  { "engine": "qm", "fastAnalysisSeconds": 20 }

How to change the defaults in ave_analysis
- Open src\main.cpp and locate the BPM config block:
  pipeline->setModuleConfig("BPM", {
    {"minBPM", 20},
    {"maxBPM", 220},
    {"frameSize", 1024},
    {"hopSize", 512},
    {"acfWindowSec", 8.0},
    {"historySize", 10},
    {"octaveCorrection", true},
    {"engine", "qm"}
  });
- Edit values as needed, rebuild, and rerun. To use the native engine, set {"engine","native"}.

Other module configuration
- Onset
  - sensitivity (0..1): detection sensitivity, default 0.5
- Spectral
  - fftSize: FFT size for spectral features (e.g., 2048)
  - hopSize: hop size for spectral features (e.g., 512)
- Structure
  - segmentMinLength (seconds): minimum segment duration, e.g., 8.0
- Cue
  - anticipationTime (seconds): how far ahead to place cues before events, e.g., 1.5

Environment variables (Audio loader fallbacks)
- If loading the WAV fails, ave_analysis will create a silent buffer using:
  - AVE_FALLBACK_DURATION_SEC (default 10)
  - AVE_FALLBACK_CHANNELS (default 2)
  - AVE_FALLBACK_SR (default 44100)

Output overview (analysis.json)
- tempo:
  - bpm (number): estimated BPM (non‑integer values reflect precise grid)
  - confidence (0..1)
  - beatGrid: array of beats as objects { t, strength }
  - downbeats: array of downbeat timestamps (every 4th beat)
- features:
  - onsets: compact onset list
  - bands/spectralInfo: spectral features
- structure: segment array (prefer enhanced segments from Cue when available)
- cues: array of cue points

Tips & troubleshooting
- I still get Mixxx‑different BPM:
  - Ensure engine is set to "qm" in the BPM config. This routes analysis through QM‑DSP DetectionFunction + TempoTrackV2.
- I want integer BPM:
  - The tool reports high‑precision BPM from the beat grid. If you need integers, round tempo["bpm"] in your consumer.
- Faster runs for batch processing:
  - Use fastAnalysisSeconds (10–45) and keep engine="qm" for robust estimates.
- Sample rate invariance:
  - QM‑DSP uses hop≈0.01161 s and window≈nextPow2(fs/50) internally; results are robust across 44.1/48/96 kHz.

Developer notes
- Library targets:
  - ave_core: core analysis library (link this in your app)
  - ave_qmdsp: bundled QM‑DSP static library (DetectionFunction, TempoTrackV2, FFT via kissfft in double precision)
- Executables:
  - ave_analysis: CLI tool used above
  - ave_tests: unit/integration tests
- Build dependencies bundled: nlohmann/json (FetchContent), FFTW3 DLL provided in fftw-3.3.5-dll64 for Windows runtime.

License notes
- The integrated QM‑DSP sources are licensed under GPL as provided in mixxx/lib/qm-dsp. Ensure your usage complies with their license when redistributing.
