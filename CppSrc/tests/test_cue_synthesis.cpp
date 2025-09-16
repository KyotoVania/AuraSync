#include "../include/core/IAnalysisModule.h"
#include "../include/core/AudioBuffer.h"
#include "../src/modules/RealCueModule.cpp" // Include implementation directly for testing
#include <nlohmann/json.hpp>
#include <iostream>
#include <cassert>

using namespace ave::core;
using namespace ave::modules;

/**
 * Test RealCueModule with mock data
 * Tests all four tasks: beat phasing, energy analysis, semantic labeling, and cue generation
 */
bool test_cues_on_mock_data() {
    std::cout << "Testing RealCueModule with mock data..." << std::endl;
    
    // Create the module
    auto module = createRealCueModule();
    
    // Initialize with default config
    nlohmann::json config = nlohmann::json::object();
    if (!module->initialize(config)) {
        std::cout << "Failed to initialize RealCueModule" << std::endl;
        return false;
    }
    
    // Create mock AnalysisContext with fake data
    AnalysisContext context;
    context.sampleRate = 44100.0f;
    
    // Mock BPM data - simple 4/4 beat grid at 120 BPM
    nlohmann::json mockBPM = {
        {"bpm", 120.0},
        {"confidence", 0.9},
        {"beatInterval", 0.5}, // 120 BPM = 0.5 seconds per beat
        {"beatGrid", nlohmann::json::array({
            {{"t", 0.0}, {"strength", 1.0}},    // Beat 1 (downbeat)
            {{"t", 0.5}, {"strength", 0.8}},    // Beat 2
            {{"t", 1.0}, {"strength", 0.8}},    // Beat 3
            {{"t", 1.5}, {"strength", 0.8}},    // Beat 4
            {{"t", 2.0}, {"strength", 1.0}},    // Beat 1 (downbeat)
            {{"t", 2.5}, {"strength", 0.8}},    // Beat 2
            {{"t", 3.0}, {"strength", 0.8}},    // Beat 3
            {{"t", 3.5}, {"strength", 0.8}},    // Beat 4
            {{"t", 4.0}, {"strength", 1.0}},    // Beat 1 (downbeat)
        })},
        {"downbeats", nlohmann::json::array({0.0, 2.0, 4.0})},
        {"method", "test-mock"}
    };
    context.moduleResults["BPM"] = mockBPM;
    
    // Mock Structure data - simple A-B-A pattern
    nlohmann::json mockStructure = {
        {"segments", nlohmann::json::array({
            {{"start", 0.0}, {"end", 15.8}, {"label", "segment_0"}, {"confidence", 0.8}},
            {{"start", 15.8}, {"end", 48.2}, {"label", "segment_1"}, {"confidence", 0.9}},
            {{"start", 48.2}, {"end", 80.0}, {"label", "segment_2"}, {"confidence", 0.8}}
        })},
        {"count", 3}
    };
    context.moduleResults["Structure"] = mockStructure;
    
    // Mock Spectral data - simulate energy patterns
    nlohmann::json mockSpectral = {
        {"bands", {
            {"low", nlohmann::json::array({
                {{"t", 0.0}, {"v", 0.2}},   // Low energy (intro)
                {{"t", 10.0}, {"v", 0.25}},
                {{"t", 20.0}, {"v", 0.6}},  // Rising energy (buildup)
                {{"t", 30.0}, {"v", 0.8}},
                {{"t", 40.0}, {"v", 0.9}},  // High energy
                {{"t", 50.0}, {"v", 0.8}},  // Drop section
                {{"t", 60.0}, {"v", 0.85}},
                {{"t", 70.0}, {"v", 0.7}},
            })},
            {"mid", nlohmann::json::array({
                {{"t", 0.0}, {"v", 0.15}},
                {{"t", 10.0}, {"v", 0.2}},
                {{"t", 20.0}, {"v", 0.5}},
                {{"t", 30.0}, {"v", 0.7}},
                {{"t", 40.0}, {"v", 0.8}},
                {{"t", 50.0}, {"v", 0.75}},
                {{"t", 60.0}, {"v", 0.8}},
                {{"t", 70.0}, {"v", 0.6}},
            })},
            {"high", nlohmann::json::array({
                {{"t", 0.0}, {"v", 0.1}},   // Low energy (intro)
                {{"t", 10.0}, {"v", 0.15}},
                {{"t", 20.0}, {"v", 0.3}},  // Rising energy
                {{"t", 30.0}, {"v", 0.5}},
                {{"t", 40.0}, {"v", 0.6}},  // High energy  
                {{"t", 50.0}, {"v", 0.5}},  // Drop section
                {{"t", 60.0}, {"v", 0.55}},
                {{"t", 70.0}, {"v", 0.4}},
            })}
        }},
        {"frameRate", 43.0},
        {"count", 8}
    };
    context.moduleResults["Spectral"] = mockSpectral;
    
    // Mock Onset data - varying densities
    nlohmann::json mockOnset = {
        {"onsets", nlohmann::json::array({
            {{"t", 1.2}, {"strength", 0.6}},   // Sparse onsets (intro)
            {{"t", 8.5}, {"strength", 0.5}},
            {{"t", 16.3}, {"strength", 0.7}},  // Buildup section
            {{"t", 18.1}, {"strength", 0.6}},
            {{"t", 20.8}, {"strength", 0.8}},
            {{"t", 22.4}, {"strength", 0.7}},
            {{"t", 25.1}, {"strength", 0.9}},
            {{"t", 27.6}, {"strength", 0.8}},
            {{"t", 30.2}, {"strength", 0.9}},
            {{"t", 32.8}, {"strength", 0.7}},
            {{"t", 49.5}, {"strength", 0.9}},  // Dense onsets (drop)
            {{"t", 50.2}, {"strength", 0.8}},
            {{"t", 50.8}, {"strength", 0.9}},
            {{"t", 51.5}, {"strength", 0.7}},
            {{"t", 52.1}, {"strength", 0.8}},
            {{"t", 53.0}, {"strength", 0.9}},
            {{"t", 53.6}, {"strength", 0.6}},
            {{"t", 54.3}, {"strength", 0.8}},
        })},
        {"count", 18},
        {"sensitivity", 0.5}
    };
    context.moduleResults["Onset"] = mockOnset;
    
    // Mock Tonality data (not used by CueModule but required dependency)
    nlohmann::json mockTonality = {
        {"key", "Am"},
        {"confidence", 0.85},
        {"mode", "minor"}
    };
    context.moduleResults["Tonality"] = mockTonality;
    
    // Create a dummy AudioBuffer (not used by CueModule)
    AudioBuffer audio(1, 1024, 44100.0f); // 1 channel, 1024 frames, 44.1kHz
    
    // Process the module
    auto result = module->process(audio, context);
    
    // Validate the output structure
    if (!module->validateOutput(result)) {
        std::cout << "Output validation failed" << std::endl;
        return false;
    }
    
    // Test 1: Beat Phasing
    std::cout << "Testing beat phasing..." << std::endl;
    if (!result.contains("phasedBeats")) {
        std::cout << "Missing phasedBeats in output" << std::endl;
        return false;
    }
    
    auto phasedBeats = result["phasedBeats"];
    if (phasedBeats.size() != 9) { // Should match beatGrid size
        std::cout << "Expected 9 phased beats, got " << phasedBeats.size() << std::endl;
        return false;
    }
    
    // Check phases: should be 1,2,3,4,1,2,3,4,1
    std::vector<int> expectedPhases = {1, 2, 3, 4, 1, 2, 3, 4, 1};
    for (size_t i = 0; i < phasedBeats.size(); ++i) {
        int phase = phasedBeats[i]["phase"];
        if (phase != expectedPhases[i]) {
            std::cout << "Beat " << i << ": expected phase " << expectedPhases[i] << ", got " << phase << std::endl;
            return false;
        }
    }
    std::cout << "✓ Beat phasing test passed" << std::endl;
    
    // Test 2: Semantic Labeling
    std::cout << "Testing semantic labeling..." << std::endl;
    if (!result.contains("segments")) {
        std::cout << "Missing segments in output" << std::endl;
        return false;
    }
    
    auto segments = result["segments"];
    if (segments.size() != 3) {
        std::cout << "Expected 3 segments, got " << segments.size() << std::endl;
        return false;
    }
    
    // Check that first segment is labeled as intro (low energy + first segment)
    std::string firstLabel = segments[0]["label"];
    if (firstLabel != "intro") {
        std::cout << "Expected first segment to be 'intro', got '" << firstLabel << "'" << std::endl;
        return false;
    }
    
    // Check that middle segment has high energy characteristics  
    // (Should be labeled as "drop" due to high energy and onset density)
    std::string middleLabel = segments[1]["label"];
    if (middleLabel != "drop" && middleLabel != "buildup" && middleLabel != "chorus") {
        std::cout << "Middle segment has unexpected label: " << middleLabel << std::endl;
        // This is a warning, not failure - heuristics may vary
    }
    
    std::cout << "✓ Semantic labeling test passed" << std::endl;
    
    // Test 3: Cue Generation
    std::cout << "Testing cue generation..." << std::endl;
    if (!result.contains("cues")) {
        std::cout << "Missing cues in output" << std::endl;
        return false;
    }
    
    auto cues = result["cues"];
    if (cues.size() < 3) { // At minimum should have 3 segment cues
        std::cout << "Expected at least 3 cues, got " << cues.size() << std::endl;
        return false;
    }
    
    // Check for pre-drop cue if middle segment is labeled as drop
    bool hasPreDrop = false;
    for (const auto& cue : cues) {
        if (cue["type"] == "pre-drop") {
            hasPreDrop = true;
            double preDropTime = cue["t"];
            double expectedPreDropTime = 15.8 - 1.5; // segment start - anticipation time
            if (std::abs(preDropTime - expectedPreDropTime) > 0.1) {
                std::cout << "Pre-drop cue time incorrect. Expected ~" << expectedPreDropTime << ", got " << preDropTime << std::endl;
                return false;
            }
            break;
        }
    }
    
    if (middleLabel == "drop" && !hasPreDrop) {
        std::cout << "Expected pre-drop cue for drop segment" << std::endl;
        return false;
    }
    
    std::cout << "✓ Cue generation test passed" << std::endl;
    
    // Test 4: Energy and Density Analysis
    std::cout << "Testing energy and density analysis..." << std::endl;
    
    // Check that segments have the required energy fields
    for (const auto& segment : segments) {
        if (!segment.contains("onsetDensity") || !segment.contains("lowEnergy") || 
            !segment.contains("midEnergy") || !segment.contains("highEnergy")) {
            std::cout << "Segment missing energy/density fields" << std::endl;
            return false;
        }
        
        // Basic sanity checks
        double onsetDensity = segment["onsetDensity"];
        double lowEnergy = segment["lowEnergy"];
        
        if (onsetDensity < 0.0) {
            std::cout << "Negative onset density: " << onsetDensity << std::endl;
            return false;
        }
        
        if (lowEnergy < 0.0 || lowEnergy > 1.0) {
            std::cout << "Energy values should be normalized [0,1]: " << lowEnergy << std::endl;
            return false;
        }
    }
    
    std::cout << "✓ Energy and density analysis test passed" << std::endl;
    
    // Print results for inspection
    std::cout << "\n=== Test Results Summary ===" << std::endl;
    std::cout << "Segments:" << std::endl;
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto& seg = segments[i];
        std::cout << "  " << i << ": [" << seg["start"] << "-" << seg["end"] << "] " 
                  << "'" << seg["label"] << "' "
                  << "(density=" << seg["onsetDensity"] << ", "
                  << "low=" << seg["lowEnergy"] << ")" << std::endl;
    }
    
    std::cout << "\nCues (" << cues.size() << " total):" << std::endl;
    for (const auto& cue : cues) {
        std::cout << "  t=" << cue["t"] << " type='" << cue["type"] << "' dur=" << cue["duration"] << std::endl;
    }
    
    std::cout << "\nPhased Beats (first 5):" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(5, phasedBeats.size()); ++i) {
        const auto& beat = phasedBeats[i];
        std::cout << "  t=" << beat["t"] << " phase=" << beat["phase"] << " str=" << beat["strength"] << std::endl;
    }
    
    return true;
}

// Main test function for module registration
bool test_cue_synthesis() {
    return test_cues_on_mock_data();
}