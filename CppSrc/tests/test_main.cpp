//
// Created by jeanc on 8/27/2025.
//
#include <iostream>

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

    std::cout << "Running tests..." << std::endl;

    std::cout << "- test_bpm_on_clicktrack: ";
    if (test_bpm_on_clicktrack()) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        failed++;
    }

    std::cout << "- test_spectral_on_sine_1000hz: ";
    if (test_spectral_on_sine_1000hz()) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        failed++;
    }

    std::cout << "- test_spectral_on_white_noise: ";
    if (test_spectral_on_white_noise()) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        failed++;
    }

    std::cout << "- test_onset_on_clicktrack: ";
    if (test_onset_on_clicktrack()) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        failed++;
    }

    std::cout << "- test_onset_on_silence: ";
    if (test_onset_on_silence()) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        failed++;
    }

    std::cout << "- test_onset_on_ramp: ";
    if (test_onset_on_ramp()) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        failed++;
    }

    std::cout << "- test_onset_on_clicktrack_90bpm: ";
    if (test_onset_on_clicktrack_90bpm()) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        failed++;
    }

    std::cout << "- test_onset_on_accented_clicks: ";
    if (test_onset_on_accented_clicks()) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        failed++;
    }

    std::cout << "- test_onset_on_noise_bursts: ";
    if (test_onset_on_noise_bursts()) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        failed++;
    }

    std::cout << "- test_tonality_chroma_on_a440: ";
    if (test_tonality_chroma_on_a440()) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        failed++;
    }

    std::cout << "- test_tonality_on_cmajor_scale: ";
    if (test_tonality_on_cmajor_scale()) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        failed++;
    }

    std::cout << "- test_tonality_on_aminor_chord: ";
    if (test_tonality_on_aminor_chord()) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        failed++;
    }

    if (failed == 0) {
        std::cout << "All tests passed" << std::endl;
        return 0;
    } else {
        std::cout << failed << " test(s) failed" << std::endl;
        return 1;
    }
}