//
// Created by jeanc on 8/27/2025.
//
#include <iostream>

// Declarations of test functions
bool test_bpm_on_clicktrack();
bool test_spectral_on_sine_1000hz();
bool test_spectral_on_white_noise();

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

    if (failed == 0) {
        std::cout << "All tests passed" << std::endl;
        return 0;
    } else {
        std::cout << failed << " test(s) failed" << std::endl;
        return 1;
    }
}