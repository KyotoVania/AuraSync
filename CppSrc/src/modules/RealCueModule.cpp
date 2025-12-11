#include "../../include/core/IAnalysisModule.h"
#include "../../include/core/AudioBuffer.h"
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <map>
#include <limits>
#include <cctype>

namespace ave::modules {

/**
 * Real Cue Module - Final synthesis module that combines results from all other modules
 * to generate high-level semantic cues and enhanced segment labeling.
 */
class RealCueModule : public core::IAnalysisModule {
private:
    double m_anticipationTime = 1.5; // seconds for pre-drop cues
    double m_formalSimilarityThreshold = 0.85; // threshold for segment clustering
    double m_consensusThreshold = 0.70; // threshold for consensus refinement on functional labels
    
public:
    std::string getName() const override { return "Cue"; }
    std::string getVersion() const override { return "1.0.0-real"; }
    
    std::vector<std::string> getDependencies() const override {
        return {"BPM", "Onset", "Spectral", "Tonality", "Structure"};
    }
    
    bool initialize(const nlohmann::json& config) override {
        if (config.contains("anticipationTime")) {
            m_anticipationTime = config["anticipationTime"];
        }
        if (config.contains("formalSimilarityThreshold")) {
            m_formalSimilarityThreshold = std::clamp(config["formalSimilarityThreshold"].get<double>(), 0.0, 1.0);
        }
        if (config.contains("consensusThreshold")) {
            m_consensusThreshold = std::clamp(config["consensusThreshold"].get<double>(), 0.0, 1.0);
        }
        return true;
    }
    
    void reset() override {
        m_anticipationTime = 1.5;
    }
    
    nlohmann::json process(const core::AudioBuffer& audio, 
                           const core::AnalysisContext& context) override {
        // Task 1: Beat Phasing
        auto phasedBeats = createPhasedBeats(context);
        
        // Task 2: Analyze energy and density per segment
        auto enrichedSegments = analyzeSegmentMetrics(context);
        
        // Task 3: Apply semantic labeling to segments
        auto labeledSegments = applySemanticLabeling(enrichedSegments);
        
        // Task 3b: Build continuous intensity curve from segment energies
        auto intensityCurve = buildIntensityCurve(labeledSegments);
        
        // Task 4: Generate anticipation cues
        auto cues = generateCues(labeledSegments);
        
        return {
            {"segments", labeledSegments},
            {"phasedBeats", phasedBeats},
            {"cues", cues},
            {"intensityCurve", intensityCurve}
        };
    }
    
    bool validateOutput(const nlohmann::json& output) const override {
        return output.contains("segments") && 
               output.contains("phasedBeats") && 
               output.contains("cues");
    }
    
private:
    /**
     * Task 1: Create phased beats from BPM data
     * Adds phase information (1,2,3,4) to beat grid based on downbeats
     */
    nlohmann::json createPhasedBeats(const core::AnalysisContext& context) {
        auto bpmResult = context.getModuleResult("BPM");
        if (!bpmResult || !bpmResult->contains("beatGrid") || !bpmResult->contains("downbeats")) {
            return nlohmann::json::array();
        }
        
        auto& beatGrid = (*bpmResult)["beatGrid"];
        auto& downbeats = (*bpmResult)["downbeats"];
        
        nlohmann::json phasedBeats = nlohmann::json::array();
        int beatPhase = 1;
        size_t downbeatIndex = 0;
        
        for (const auto& beat : beatGrid) {
            float beatTime = beat["t"];
            
            // Check if this beat is a downbeat (within small tolerance)
            bool isDownbeat = false;
            while (downbeatIndex < downbeats.size()) {
                float downbeatTime = downbeats[downbeatIndex];
                if (std::abs(beatTime - downbeatTime) < 0.05f) { // 50ms tolerance
                    beatPhase = 1;
                    isDownbeat = true;
                    downbeatIndex++;
                    break;
                } else if (downbeatTime > beatTime) {
                    break;
                } else {
                    downbeatIndex++;
                }
            }
            
            phasedBeats.push_back({
                {"t", beatTime},
                {"strength", beat["strength"]},
                {"phase", beatPhase}
            });
            
            // Increment phase for next beat (unless this was a downbeat that resets to 1)
            if (!isDownbeat) {
                beatPhase = (beatPhase % 4) + 1;
            } else {
                // After a downbeat (phase 1), next beat should be phase 2
                beatPhase = 2;
            }
        }
        
        return phasedBeats;
    }
    
    /**
     * Task 2: Analyze energy and density metrics for each segment
     */
    // Feature container for rule-based scoring
    struct SegmentFeatures {
        double duration = 0.0;
        // Rhythmic
        double onsetDensity = 0.0;
        // Spectral/timbral
        double lowEnergy = 0.0, midEnergy = 0.0, highEnergy = 0.0;
        double bassEnergy = 0.0; // combined low/sub energy for dynamic bands
        double spectralCentroidMean = 0.0, spectralCentroidStdDev = 0.0;
        // Harmonic
        double keyClarity = 0.0;
        // Derived convenience
        double overallEnergy = 0.0;
        // New features for richer semantics
        double timbralStability = 0.0;  // computed as 1 - normalized MFCC variance
        double rhythmStability = 0.0;   // computed as 1 - normalized IOI variance
        double relativePosition = 0.0;  // segment mid-time / track duration
    };

    static double overallEnergyFromBands(double low, double mid, double high) {
        // Emphasize bass slightly
        return 0.5 * low + 0.3 * mid + 0.2 * high;
    }

    nlohmann::json analyzeSegmentMetrics(const core::AnalysisContext& context) {
        auto structureResult = context.getModuleResult("Structure");
        auto spectralResult = context.getModuleResult("Spectral");
        auto onsetResult = context.getModuleResult("Onset");
        auto tonalityResult = context.getModuleResult("Tonality");
        
        if (!structureResult || !structureResult->contains("segments")) {
            return nlohmann::json::array();
        }
        
        auto segments = (*structureResult)["segments"];
        nlohmann::json enrichedSegments = nlohmann::json::array();
        
        // Key clarity from tonality (track-level confidence) if available
        double keyClarityTrack = 0.0;
        if (tonalityResult && tonalityResult->contains("confidence")) {
            keyClarityTrack = (*tonalityResult)["confidence"].get<double>();
        }

        // Track duration from last segment end, if available
        double trackDuration = 0.0;
        if (segments.is_array() && !segments.empty()) {
            const auto& lastSeg = segments.back();
            if (lastSeg.contains("end")) trackDuration = lastSeg["end"].get<double>();
        }
        if (trackDuration <= 0.0) trackDuration = 1.0; // avoid division by zero

        for (const auto& segment : segments) {
            double start = segment["start"].get<double>();
            double end = segment["end"].get<double>();
            double duration = std::max(0.0, end - start);

            // Calculate onset density
            double onsetDensity = calculateOnsetDensity(onsetResult, start, end);

            // Calculate spectral energy (band averages)
            auto energies = calculateSpectralEnergies(spectralResult, start, end);

            // Dynamic spectral centroid using all available bands
            double centroidSum = 0.0, centroidSqSum = 0.0; int centroidCount = 0;
            if (spectralResult && spectralResult->contains("bands")) {
                const auto& bands = (*spectralResult)["bands"];
                if (bands.is_object() && !bands.empty()) {
                    // Collect band names and define approximate center frequencies by name
                    std::vector<std::string> bandNames; bandNames.reserve(bands.size());
                    for (auto it = bands.begin(); it != bands.end(); ++it) {
                        if (it.value().is_array()) bandNames.push_back(it.key());
                    }
                    // Build center frequency lookup
                    auto toLower = [](std::string s){ std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); }); return s; };
                    std::map<std::string,double> nameCenter = {
                        {"sub",60.0}, {"low",150.0}, {"lowmid",400.0}, {"low_mid",400.0},
                        {"mid",1000.0}, {"highmid",3000.0}, {"high_mid",3000.0}, {"high",8000.0}, {"air",12000.0}
                    };
                    std::vector<double> centers; centers.reserve(bandNames.size());
                    const double fLoDefault = 80.0, fHiDefault = 15000.0;
                    for (size_t bi = 0; bi < bandNames.size(); ++bi) {
                        std::string ln = toLower(bandNames[bi]);
                        auto itc = nameCenter.find(ln);
                        if (itc != nameCenter.end()) centers.push_back(itc->second);
                        else centers.push_back(fLoDefault + (fHiDefault - fLoDefault) * ((static_cast<double>(bi)+0.5) / static_cast<double>(std::max<size_t>(1, bandNames.size()))));
                    }
                    // Determine consistent frame count (min across bands)
                    size_t N = std::numeric_limits<size_t>::max();
                    for (const auto& nm : bandNames) { N = std::min(N, bands[nm].size()); }
                    if (N == std::numeric_limits<size_t>::max()) N = 0;
                    if (N > 0) {
                        // Use first band as reference for timestamps
                        const auto& ref = bands[bandNames.front()];
                        for (size_t i = 0; i < N; ++i) {
                            if (!ref[i].contains("t")) continue;
                            double t = ref[i]["t"].get<double>();
                            if (t < start || t > end) continue;
                            double sumE = 0.0, sumEf = 0.0;
                            for (size_t bi = 0; bi < bandNames.size(); ++bi) {
                                const auto& fr = bands[bandNames[bi]][i];
                                if (!fr.contains("v")) continue;
                                double e = fr["v"].get<double>();
                                sumE += e; sumEf += e * centers[bi];
                            }
                            if (sumE > 0.0) {
                                double c = sumEf / sumE;
                                centroidSum += c; centroidSqSum += c * c; ++centroidCount;
                            }
                        }
                    }
                }
            }
            double centroidMean = (centroidCount ? centroidSum / centroidCount : 0.0);
            double centroidVar = (centroidCount ? (centroidSqSum / centroidCount - centroidMean * centroidMean) : 0.0);
            if (centroidVar < 0.0) centroidVar = 0.0;
            double centroidStd = std::sqrt(centroidVar);

            // Compute overall energy as mean across all bands
            double overallEnergy = 0.0; {
                double s = 0.0; size_t c = 0; for (const auto& kv : energies) { s += kv.second; ++c; }
                overallEnergy = (c ? (s / static_cast<double>(c)) : 0.0);
            }

            // Derive classic band values for backward-compatibility and bassEnergy
            double lowEnergy = 0.0, midEnergy = 0.0, highEnergy = 0.0, bassEnergy = 0.0;
            if (energies.count("low")) lowEnergy = energies["low"];
            if (energies.count("mid")) midEnergy = energies["mid"];
            if (energies.count("high")) highEnergy = energies["high"];
            if (energies.count("sub")) bassEnergy += energies["sub"];
            if (energies.count("low")) bassEnergy += energies["low"];
            if (bassEnergy <= 0.0 && !energies.empty()) {
                // Fallback: take the smallest-index band as bass proxy
                bassEnergy = energies.begin()->second;
            }

            // Timbral variance from MFCCs within segment
            double timbralVar = 0.0;
            if (spectralResult && spectralResult->contains("mfcc")) {
                const auto& mfcc = (*spectralResult)["mfcc"];
                // Determine number of coefficients from first frame
                size_t C = 0; if (!mfcc.empty() && mfcc[0].contains("v") && mfcc[0]["v"].is_array()) C = mfcc[0]["v"].size();
                if (C > 0) {
                    std::vector<double> sum(C, 0.0), sum2(C, 0.0);
                    size_t count = 0;
                    for (const auto& fr : mfcc) {
                        if (!fr.contains("t") || !fr.contains("v")) continue;
                        double t = fr["t"].get<double>();
                        if (t < start || t > end) continue;
                        const auto& v = fr["v"];
                        for (size_t c = 0; c < C && c < v.size(); ++c) {
                            double x = v[c].get<double>();
                            sum[c] += x; sum2[c] += x * x;
                        }
                        ++count;
                    }
                    if (count > 1) {
                        double accVar = 0.0;
                        for (size_t c = 0; c < C; ++c) {
                            double m = sum[c] / static_cast<double>(count);
                            double var = std::max(0.0, sum2[c] / static_cast<double>(count) - m * m);
                            accVar += var;
                        }
                        timbralVar = accVar / static_cast<double>(C);
                    }
                }
            }

            // Rhythm variance from inter-onset intervals within segment (coefficient of variation)
            double rhythmVar = 0.0;
            if (onsetResult && onsetResult->contains("onsets")) {
                std::vector<double> times; times.reserve(64);
                for (const auto& o : (*onsetResult)["onsets"]) {
                    if (!o.contains("t")) continue; double t = o["t"].get<double>();
                    if (t >= start && t <= end) times.push_back(t);
                }
                std::sort(times.begin(), times.end());
                if (times.size() >= 3) {
                    std::vector<double> iois; iois.reserve(times.size()-1);
                    for (size_t k = 1; k < times.size(); ++k) iois.push_back(times[k] - times[k-1]);
                    double m = 0.0, s2 = 0.0; size_t n = iois.size();
                    for (double x : iois) { m += x; s2 += x * x; }
                    m /= static_cast<double>(n);
                    double var = std::max(0.0, s2 / static_cast<double>(n) - m * m);
                    double stdv = std::sqrt(var);
                    rhythmVar = (m > 1e-6) ? (stdv / m) : 0.0; // coefficient of variation
                } else {
                    rhythmVar = 0.0; // treat sparse as stable
                }
            }

            // Relative position of segment in track [0..1]
            double mid = 0.5 * (start + end);
            double relativePosition = std::clamp(trackDuration > 0.0 ? (mid / trackDuration) : 0.0, 0.0, 1.0);

            // Create enriched segment with features
            SegmentFeatures fts;
            fts.duration = duration;
            fts.onsetDensity = onsetDensity;
            fts.lowEnergy = lowEnergy;
            fts.midEnergy = midEnergy;
            fts.highEnergy = highEnergy;
            fts.bassEnergy = bassEnergy;
            fts.spectralCentroidMean = centroidMean;
            fts.spectralCentroidStdDev = centroidStd;
            fts.keyClarity = keyClarityTrack;
            fts.overallEnergy = overallEnergy;
            fts.timbralStability = timbralVar; // temporary store raw variance; normalized later
            fts.rhythmStability = rhythmVar;   // temporary store raw coefficient; normalized later
            fts.relativePosition = relativePosition;

            nlohmann::json enriched = segment;
            enriched["duration"] = fts.duration;
            enriched["onsetDensity"] = fts.onsetDensity;
            enriched["lowEnergy"] = fts.lowEnergy;
            enriched["midEnergy"] = fts.midEnergy;
            enriched["highEnergy"] = fts.highEnergy;
            enriched["bassEnergy"] = fts.bassEnergy;
            // Also expose per-band energies for transparency
            {
                nlohmann::json bandE = nlohmann::json::object();
                for (const auto& kv : energies) bandE[kv.first] = kv.second;
                enriched["bandEnergies"] = bandE;
            }
            enriched["spectralCentroidMean"] = fts.spectralCentroidMean;
            enriched["spectralCentroidStdDev"] = fts.spectralCentroidStdDev;
            enriched["keyClarity"] = fts.keyClarity;
            enriched["overallEnergy"] = fts.overallEnergy;
            enriched["timbralVar"] = timbralVar;
            enriched["rhythmVar"] = rhythmVar;
            enriched["relativePosition"] = relativePosition;

            enrichedSegments.push_back(enriched);
        }

        return enrichedSegments;
    }

    /**
     * Task 3: Apply semantic labeling heuristics
     */
    nlohmann::json applySemanticLabeling(const nlohmann::json& enrichedSegments) {
        // Build SegmentFeatures list and compute normalization ranges
        struct Range { double minv=1e9, maxv=-1e9; void add(double x){ if(x<minv)minv=x; if(x>maxv)maxv=x; } double norm(double x) const { if(maxv<=minv) return 0.0; double y=(x-minv)/(maxv-minv); if(y<0.0) y=0.0; if(y>1.0) y=1.0; return y; } };
        std::vector<SegmentFeatures> feats;
        feats.reserve(enrichedSegments.size());
        Range rOnset, rLow, rMid, rHigh, rBass, rOverall, rDur, rCentroid, rTimbralVar, rRhythmVar;
        for (const auto& seg : enrichedSegments) {
            SegmentFeatures f{};
            f.duration = seg.value("duration", 0.0);
            f.onsetDensity = seg.value("onsetDensity", 0.0);
            f.lowEnergy = seg.value("lowEnergy", 0.0);
            f.midEnergy = seg.value("midEnergy", 0.0);
            f.highEnergy = seg.value("highEnergy", 0.0);
            f.bassEnergy = seg.value("bassEnergy", f.lowEnergy);
            f.spectralCentroidMean = seg.value("spectralCentroidMean", 0.0);
            f.spectralCentroidStdDev = seg.value("spectralCentroidStdDev", 0.0);
            f.keyClarity = seg.value("keyClarity", 0.0);
            f.overallEnergy = seg.value("overallEnergy", overallEnergyFromBands(f.lowEnergy, f.midEnergy, f.highEnergy));
            // Temporarily store raw variances in stability fields; convert below
            f.timbralStability = seg.value("timbralVar", 0.0);
            f.rhythmStability = seg.value("rhythmVar", 0.0);
            f.relativePosition = seg.value("relativePosition", 0.0);
            feats.push_back(f);
            rOnset.add(f.onsetDensity); rLow.add(f.lowEnergy); rMid.add(f.midEnergy); rHigh.add(f.highEnergy); rBass.add(f.bassEnergy); rOverall.add(f.overallEnergy); rDur.add(f.duration); rCentroid.add(f.spectralCentroidMean); rTimbralVar.add(f.timbralStability); rRhythmVar.add(f.rhythmStability);
        }

        // Convert raw variance to stability [0..1]
        for (auto& f : feats) {
            double timbralVarN = rTimbralVar.norm(f.timbralStability);
            double rhythmVarN = rRhythmVar.norm(f.rhythmStability);
            f.timbralStability = 1.0 - timbralVarN;
            f.rhythmStability = 1.0 - rhythmVarN;
        }

        // Scoring functions (all metrics expected normalized in [0,1])
        auto calculateIntroScore = [&](const SegmentFeatures& f, bool isFirst){
            double onset = 1.0 - rOnset.norm(f.onsetDensity);
            double energy = 1.0 - rOverall.norm(f.overallEnergy);
            double dur = rDur.norm(f.duration);
            double bonus = isFirst ? 0.2 : 0.0;
            return 0.4*onset + 0.4*energy + 0.2*dur + bonus;
        };
        auto calculateDropScore = [&](const SegmentFeatures& cur, const SegmentFeatures* prev){
            // Use combined bass energy (sub + low) when available
            double bass = rBass.norm(cur.bassEnergy);
            double onset = rOnset.norm(cur.onsetDensity); // active
            double inc = 0.0;
            if (prev) {
                double d = rOverall.norm(cur.overallEnergy) - rOverall.norm(prev->overallEnergy);
                inc = std::clamp(d, 0.0, 1.0);
            }
            return 0.4*bass + 0.2*onset + 0.4*inc;
        };
        auto calculateBreakdownScore = [&](const SegmentFeatures& f){
            double low = 1.0 - rLow.norm(f.lowEnergy);
            double mid = 1.0 - rMid.norm(f.midEnergy);
            double high = 1.0 - rHigh.norm(f.highEnergy);
            double onset = 1.0 - rOnset.norm(f.onsetDensity);
            return 0.25*low + 0.25*mid + 0.25*high + 0.25*onset;
        };
        auto calculateVerseScore = [&](const SegmentFeatures& f){
            double energy = 1.0 - std::abs(rOverall.norm(f.overallEnergy) - 0.4);
            double onset = 1.0 - rOnset.norm(f.onsetDensity);
            double timbre = f.timbralStability;
            return 0.4*energy + 0.3*onset + 0.3*timbre;
        };
        auto calculateChorusScore = [&](const SegmentFeatures& f){
            double energy = rOverall.norm(f.overallEnergy);
            double onset = rOnset.norm(f.onsetDensity);
            double brightness = rCentroid.norm(f.spectralCentroidMean);
            return 0.5*energy + 0.3*onset + 0.2*brightness;
        };
        auto calculateBuildupScore = [&](const SegmentFeatures& cur, const SegmentFeatures* prev){
            double energyIncrease = 0.0, onsetIncrease = 0.0;
            double lowBuildup = 0.0;
            double nBassCur = rBass.norm(cur.bassEnergy);
            if (prev) {
                double nOverallCur = rOverall.norm(cur.overallEnergy);
                double nOverallPrev = rOverall.norm(prev->overallEnergy);
                double nOnsetCur = rOnset.norm(cur.onsetDensity);
                double nOnsetPrev = rOnset.norm(prev->onsetDensity);
                double nBassPrev = rBass.norm(prev->bassEnergy);
                energyIncrease = std::max(0.0, nOverallCur - nOverallPrev);
                onsetIncrease = std::max(0.0, nOnsetCur - nOnsetPrev);
                // Favor low-bass reduction or stagnation (kick drop) before the impact
                double lowDrop = std::max(0.0, nBassPrev - nBassCur); // positive if current bass < previous bass
                double lowCut = 1.0 - nBassCur; // absolute bass cut helps too
                lowBuildup = 0.7 * lowDrop + 0.3 * lowCut;
                // Penalize cases where bass increases together with onset spike (likely the drop itself)
                if ((nBassCur > nBassPrev + 0.10) && (onsetIncrease > 0.20)) {
                    lowBuildup *= 0.8;
                }
            } else {
                // No previous context: rely on absolute low cut only
                lowBuildup = 1.0 - nBassCur;
            }
            // Slightly emphasize rhythmic density growth for buildup feeling
            return 0.4*energyIncrease + 0.35*onsetIncrease + 0.25*lowBuildup;
        };
        auto calculateOutroScore = [&](const SegmentFeatures& cur, const SegmentFeatures* prev, bool isLast){
            double rel = cur.relativePosition; // already [0..1]
            double onsetLow = 1.0 - rOnset.norm(cur.onsetDensity);
            double dec = 0.0; if (prev) dec = std::max(0.0, rOverall.norm(prev->overallEnergy) - rOverall.norm(cur.overallEnergy));
            double bonus = isLast ? 0.1 : 0.0;
            return 0.5*rel + 0.3*onsetLow + 0.2*dec + bonus;
        };
        auto calculatePreChorusScore = [&](const SegmentFeatures& cur, const SegmentFeatures* next){
            double energyTarget = 1.0 - std::abs(rOverall.norm(cur.overallEnergy) - 0.6);
            double onsetTarget = 1.0 - std::abs(rOnset.norm(cur.onsetDensity) - 0.5);
            double incNext = 0.0; if (next) incNext = std::max(0.0, rOverall.norm(next->overallEnergy) - rOverall.norm(cur.overallEnergy));
            return 0.4*incNext + 0.4*energyTarget + 0.2*onsetTarget;
        };
        auto calculateBridgeScore = [&](const SegmentFeatures& cur, const SegmentFeatures* prev, const SegmentFeatures* next){
            auto nOverall = rOverall.norm(cur.overallEnergy);
            auto nOnset = rOnset.norm(cur.onsetDensity);
            auto nCent = rCentroid.norm(cur.spectralCentroidMean);
            auto neighborContrast = [&](const SegmentFeatures* other){
                if (!other) return 0.0;
                double d = 0.0; int c = 0;
                d += std::abs(nOverall - rOverall.norm(other->overallEnergy)); ++c;
                d += std::abs(nOnset - rOnset.norm(other->onsetDensity)); ++c;
                d += std::abs(nCent - rCentroid.norm(other->spectralCentroidMean)); ++c;
                // Safe division (c is guaranteed 3)
                return d / static_cast<double>(c);
            };
            double contrast = 0.5*neighborContrast(prev) + 0.5*neighborContrast(next);
            double timbreChange = 1.0 - cur.timbralStability;
            double rhythmChange = 1.0 - cur.rhythmStability;
            return 0.5*contrast + 0.3*timbreChange + 0.2*rhythmChange;
        };

        nlohmann::json labeledSegments = nlohmann::json::array();
        for (size_t i = 0; i < enrichedSegments.size(); ++i) {
            auto segment = enrichedSegments[i];
            const SegmentFeatures& cur = feats[i];
            const SegmentFeatures* prev = (i > 0) ? &feats[i-1] : nullptr;
            const SegmentFeatures* next = (i + 1 < feats.size()) ? &feats[i+1] : nullptr;

            double sIntro = calculateIntroScore(cur, i==0);
            double sDrop = calculateDropScore(cur, prev);
            double sBreak = calculateBreakdownScore(cur);
            double sVerse = calculateVerseScore(cur);
            double sChorus = calculateChorusScore(cur);
            double sBuildup = calculateBuildupScore(cur, prev);
            double sOutro = calculateOutroScore(cur, prev, i+1==feats.size());
            double sPreChorus = calculatePreChorusScore(cur, next);
            double sBridge = calculateBridgeScore(cur, prev, next);

            std::string label = "segment";
            // Choose best label
            std::vector<std::pair<std::string,double>> scores = {
                {"intro", sIntro}, {"drop", sDrop}, {"breakdown", sBreak}, {"verse", sVerse}, {"chorus", sChorus}, {"buildup", sBuildup}, {"outro", sOutro}, {"pre-chorus", sPreChorus}, {"bridge", sBridge}
            };
            double best = -1.0;
            for (const auto& kv : scores) { if (kv.second > best) { best = kv.second; label = kv.first; } }

            // First segment bias to intro if competitive
            if (i==0 && sIntro + 0.05 >= best) { label = "intro"; }

            // Attach scores for transparency (also include stabilities)
            nlohmann::json scoreObj;
            for (const auto& kv : scores) scoreObj[kv.first] = kv.second;
            scoreObj["timbralStability"] = cur.timbralStability;
            scoreObj["rhythmStability"] = cur.rhythmStability;
            segment["scores"] = scoreObj;
            segment["label"] = label;
            labeledSegments.push_back(segment);
        }

        // Relational clustering: group similar 'chorus' segments and add numbering suffix
        {
            std::vector<size_t> chorusIdxRefined;
            for (size_t i = 0; i < labeledSegments.size(); ++i) if (labeledSegments[i].value("label", std::string()) == "chorus") chorusIdxRefined.push_back(i);
            if (!chorusIdxRefined.empty()) {
                // Simple chronological numbering for now; could refine with clustering
                int count = 1;
                for (size_t id : chorusIdxRefined) {
                    labeledSegments[id]["label"] = std::string("chorus_") + std::to_string(count++);
                }
            }
        }

        // Relational analysis: compute similarity matrix over normalized feature vectors and cluster to assign formal labels (A,B,C,...)
        const size_t S = feats.size();
        if (S > 0) {
            // Build normalized feature vectors per segment
            auto buildNormVec = [&](size_t i) {
                std::vector<double> v;
                v.reserve(9);
                v.push_back(rLow.norm(feats[i].lowEnergy));
                v.push_back(rMid.norm(feats[i].midEnergy));
                v.push_back(rHigh.norm(feats[i].highEnergy));
                v.push_back(rOnset.norm(feats[i].onsetDensity));
                v.push_back(rCentroid.norm(feats[i].spectralCentroidMean));
                v.push_back(std::clamp(feats[i].timbralStability, 0.0, 1.0));
                v.push_back(std::clamp(feats[i].rhythmStability, 0.0, 1.0));
                v.push_back(rOverall.norm(feats[i].overallEnergy));
                v.push_back(rDur.norm(feats[i].duration));
                return v;
            };
            auto dist01 = [&](const std::vector<double>& a, const std::vector<double>& b){
                size_t D = std::min(a.size(), b.size()); if (D == 0) return 1.0; double s = 0.0; for (size_t k = 0; k < D; ++k) { double d = a[k] - b[k]; s += d*d; }
                double d = std::sqrt(s) / std::sqrt(static_cast<double>(D)); if (d < 0.0) d = 0.0; if (d > 1.0) d = 1.0; return d; };
            std::vector<std::vector<double>> sim(S, std::vector<double>(S, 0.0));
            std::vector<std::vector<double>> vecs; vecs.reserve(S);
            for (size_t i = 0; i < S; ++i) vecs.push_back(buildNormVec(i));
            for (size_t i = 0; i < S; ++i) {
                sim[i][i] = 1.0;
                for (size_t j = i+1; j < S; ++j) {
                    double d = dist01(vecs[i], vecs[j]);
                    double s = 1.0 - d;
                    sim[i][j] = s; sim[j][i] = s;
                }
            }
            // Threshold clustering
            std::vector<int> clusterId(S, -1);
            int clusters = 0;
            for (size_t i = 0; i < S; ++i) {
                if (clusterId[i] != -1) continue;
                int cid = clusters++;
                clusterId[i] = cid;
                for (size_t j = i + 1; j < S; ++j) {
                    if (clusterId[j] == -1 && sim[i][j] >= m_formalSimilarityThreshold) {
                        clusterId[j] = cid;
                    }
                }
            }
            // Assign formal labels 'A','B','C', ... wrap to 'Z' then 'A2', etc.
            for (size_t i = 0; i < S; ++i) {
                int cid = clusterId[i];
                std::string formal;
                if (cid < 26) {
                    formal.push_back(static_cast<char>('A' + cid));
                } else {
                    formal = std::string("A") + std::to_string(cid + 1); // rare
                }
                labeledSegments[i]["formalLabel"] = formal;
            }

            // Phase 3.1: Consensus refinement by formal group
            // Group segments by formalLabel, determine majority functional label in each group,
            // and if consensus exceeds threshold, enforce it across the group.
            std::map<std::string, std::vector<size_t>> groups;
            for (size_t i = 0; i < labeledSegments.size(); ++i) {
                std::string f = labeledSegments[i].value("formalLabel", std::string());
                groups[f].push_back(i);
            }
            auto baseFunctional = [](const std::string& L){
                // Reduce labels like "chorus_2" to base functional label "chorus"
                auto pos = L.find('_');
                return (pos == std::string::npos) ? L : L.substr(0, pos);
            };
            for (const auto& kv : groups) {
                const auto& idxs = kv.second;
                if (idxs.size() < 2) continue; // nothing to refine
                std::map<std::string, int> counts;
                for (size_t i : idxs) {
                    std::string lab = baseFunctional(labeledSegments[i].value("label", std::string("segment")));
                    counts[lab]++;
                }
                // Find majority
                std::string majLab; int majCount = 0;
                for (const auto& c : counts) { if (c.second > majCount) { majCount = c.second; majLab = c.first; } }
                double ratio = (idxs.empty() ? 0.0 : static_cast<double>(majCount) / static_cast<double>(idxs.size()));
                if (ratio >= m_consensusThreshold && !majLab.empty()) {
                    for (size_t i : idxs) labeledSegments[i]["label"] = majLab; // enforce consensus
                }
            }
            // Re-number chorus segments chronologically after consensus to keep suffixes tidy
            {
                std::vector<size_t> chorusIdxRefined;
                for (size_t i = 0; i < labeledSegments.size(); ++i) if (baseFunctional(labeledSegments[i].value("label", std::string())) == "chorus") chorusIdxRefined.push_back(i);
                int num = 1; for (size_t id : chorusIdxRefined) labeledSegments[id]["label"] = std::string("chorus_") + std::to_string(num++);
            }
        }

        return labeledSegments;
    }
    
    /**
     * Build a simple continuous intensity curve from per-segment overall energy.
     * Produces points at segment mid-times, values normalized to [0,1], with slight smoothing.
     */
    nlohmann::json buildIntensityCurve(const nlohmann::json& labeledSegments) {
        nlohmann::json curve = nlohmann::json::array();
        if (!labeledSegments.is_array() || labeledSegments.empty()) return curve;
        // Gather midpoints and energies
        std::vector<double> t; t.reserve(labeledSegments.size());
        std::vector<double> e; e.reserve(labeledSegments.size());
        double emin = 1e9, emax = -1e9;
        for (const auto& seg : labeledSegments) {
            double start = seg.value("start", 0.0);
            double end = seg.value("end", start);
            double mid = 0.5 * (start + end);
            double overall = seg.value("overallEnergy", 0.0);
            t.push_back(mid); e.push_back(overall);
            if (overall < emin) emin = overall; if (overall > emax) emax = overall;
        }
        // Normalize energies
        auto normE = [&](double x){ if (emax <= emin) return 0.0; double y = (x - emin) / (emax - emin); if (y<0.0) y=0.0; if (y>1.0) y=1.0; return y; };
        for (double& x : e) x = normE(x);
        // Smooth with small moving average
        std::vector<double> es(e.size(), 0.0);
        for (size_t i = 0; i < e.size(); ++i) {
            double acc = 0.0; int c = 0;
            for (int di = -1; di <= 1; ++di) {
                int j = static_cast<int>(i) + di; if (j < 0 || j >= static_cast<int>(e.size())) continue; acc += e[static_cast<size_t>(j)]; ++c;
            }
            es[i] = (c ? acc / c : e[i]);
        }
        for (size_t i = 0; i < t.size(); ++i) curve.push_back({{"t", t[i]}, {"v", es[i]}});
        return curve;
    }

    /**
     * Task 4: Generate anticipation cues and segment cues
     */
    nlohmann::json generateCues(const nlohmann::json& labeledSegments) {
        nlohmann::json cues = nlohmann::json::array();
        
        for (size_t i = 0; i < labeledSegments.size(); ++i) {
            const auto& segment = labeledSegments[i];
            std::string label = segment["label"];
            double start = segment["start"];
            double end = segment["end"];
            double duration = end - start;
            
            // Generate anticipation cues for drops
            if (label == "drop" && i > 0) {
                std::string prevLabel = labeledSegments[i-1]["label"];
                double preDur = m_anticipationTime;
                if (prevLabel == "buildup") {
                    double prevDur = labeledSegments[i-1].value("duration", labeledSegments[i-1]["end"].get<double>() - labeledSegments[i-1]["start"].get<double>());
                    if (prevDur > 0.1) preDur = prevDur;
                }
                double preDropTime = start - preDur;
                if (preDropTime >= 0.0) {
                    cues.push_back({
                        {"t", preDropTime},
                        {"type", "pre-drop"},
                        {"duration", preDur}
                    });
                }
            }
            
            // Generate segment cues
            cues.push_back({
                {"t", start},
                {"type", label},
                {"duration", duration}
            });
        }
        
        return cues;
    }
    
    /**
     * Helper: Calculate onset density in a time range
     */
    double calculateOnsetDensity(const std::optional<nlohmann::json>& onsetResult, 
                                double start, double end) {
        if (!onsetResult || !onsetResult->contains("onsets")) {
            return 0.0;
        }
        
        auto& onsets = (*onsetResult)["onsets"];
        int count = 0;
        
        for (const auto& onset : onsets) {
            double t = onset["t"];
            if (t >= start && t <= end) {
                count++;
            }
        }
        
        double duration = end - start;
        return duration > 0.0 ? static_cast<double>(count) / duration : 0.0;
    }
    
    /**
     * Helper: Calculate average spectral energies in a time range (dynamic bands)
     * Returns a map bandName -> average energy in [start,end].
     */
    std::map<std::string, double> calculateSpectralEnergies(const std::optional<nlohmann::json>& spectralResult,
                                                            double start, double end) {
        std::map<std::string, double> energies;
        if (!spectralResult || !spectralResult->contains("bands")) {
            return energies;
        }
        const auto& bands = (*spectralResult)["bands"];
        if (!bands.is_object()) return energies;
        for (auto it = bands.begin(); it != bands.end(); ++it) {
            const std::string bandName = it.key();
            const auto& arr = it.value();
            if (!arr.is_array()) continue;
            double avg = calculateBandAverage(arr, start, end);
            energies[bandName] = avg;
        }
        return energies;
    }
    
    /**
     * Helper: Calculate average value for a band in time range
     */
    double calculateBandAverage(const nlohmann::json& band, double start, double end) {
        if (!band.is_array()) {
            return 0.0;
        }
        
        double sum = 0.0;
        int count = 0;
        
        for (const auto& frame : band) {
            if (frame.contains("t") && frame.contains("v")) {
                double t = frame["t"];
                if (t >= start && t <= end) {
                    sum += static_cast<double>(frame["v"]);
                    count++;
                }
            }
        }
        
        return count > 0 ? sum / count : 0.0;
    }
};

std::unique_ptr<core::IAnalysisModule> createRealCueModule() {
    return std::make_unique<RealCueModule>();
}

} // namespace ave::modules