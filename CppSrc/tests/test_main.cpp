//
// Created by jeanc on 8/27/2025.
//
#include <iostream>
#include <cstdlib>

// Declarations of test functions
bool test_bpm_on_clicktrack();
bool test_spectral_on_sine_1000hz();
bool test_spectral_on_white_noise();
bool test_onset_on_clicktrack();
bool test_onset_on_silence();
bool test_onset_on_ramp();
bool test_onset_on_clicktrack_90bpm();
bool test_onset_on_accented_clicks();
bool test_onset_on_noise_bursts();
bool test_tonality_chroma_on_a440();
bool test_tonality_on_cmajor_scale();
bool test_tonality_on_aminor_chord();

int main() {
    int failed = 0;
    int total = 0;

    const char* quietEnv = std::getenv("AVE_TEST_QUIET");
    bool quiet = quietEnv && std::string(quietEnv) != "0";

    if (!quiet) std::cout << "Running tests..." << std::endl;

    // Helper macro to run a test with optional verbose output
    auto run_test = [&](const char* name, bool (*fn)()) {
        ++total;
        bool ok = fn();
        if (!quiet) {
            std::cout << "- " << name << ": " << (ok ? "PASS" : "FAIL") << std::endl;
        }
        if (!ok) ++failed;
    };

    run_test("test_bpm_on_clicktrack", &test_bpm_on_clicktrack);
    run_test("test_spectral_on_sine_1000hz", &test_spectral_on_sine_1000hz);
    run_test("test_spectral_on_white_noise", &test_spectral_on_white_noise);
    run_test("test_onset_on_clicktrack", &test_onset_on_clicktrack);
    run_test("test_onset_on_silence", &test_onset_on_silence);
    run_test("test_onset_on_ramp", &test_onset_on_ramp);
    run_test("test_onset_on_clicktrack_90bpm", &test_onset_on_clicktrack_90bpm);
    run_test("test_onset_on_accented_clicks", &test_onset_on_accented_clicks);
    run_test("test_onset_on_noise_bursts", &test_onset_on_noise_bursts);
    run_test("test_tonality_chroma_on_a440", &test_tonality_chroma_on_a440);
    run_test("test_tonality_on_cmajor_scale", &test_tonality_on_cmajor_scale);
    run_test("test_tonality_on_aminor_chord", &test_tonality_on_aminor_chord);

    int passed = total - failed;
    std::cout << "Summary: " << passed << "/" << total << " passed, " << failed << " failed" << std::endl;

    return failed == 0 ? 0 : 1;
}