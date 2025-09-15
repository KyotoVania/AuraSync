cmake_minimum_required(VERSION 3.16)
project(AudioVisualEngineAnalysis VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# --- Dependencies ---

# nlohmann/json
include(FetchContent)
FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# FFTW3 (assuming it's in fftw-3.3.5-dll64)
set(FFTW3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fftw-3.3.5-dll64)

# --- Library ---

# Core library
add_library(ave_core STATIC
        # Core
        src/core/AudioBuffer.cpp
        src/core/JsonContract.cpp

        # Pipeline
        src/pipeline/AnalysisPipeline.cpp
        src/pipeline/AudioLoader.cpp

        # Modules
        src/modules/RealBPMModule.cpp
        src/modules/RealSpectralModule.cpp
        src/modules/FakeBPMModule.cpp
        src/modules/FakeModuleOnset.cpp
)

target_include_directories(ave_core
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FFTW3_DIR}
)

target_link_libraries(ave_core
        PUBLIC
        nlohmann_json::nlohmann_json
        ${FFTW3_DIR}/libfftw3-3.lib
)

# --- Main Executable ---

add_executable(ave_analysis src/main.cpp)

target_link_libraries(ave_analysis
        PRIVATE
        ave_core
)

# Copy FFTW DLL to output directory for main
add_custom_command(TARGET ave_analysis POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${FFTW3_DIR}/libfftw3-3.dll"
        $<TARGET_FILE_DIR:ave_analysis>
)

# --- Tests ---

option(BUILD_TESTS "Build tests" ON)

if(BUILD_TESTS)
    enable_testing()

    # Test executable
    add_executable(ave_tests
            tests/test_main.cpp
            tests/test_modules.cpp
            tests/test_pipeline.cpp
            tests/test_json_contract.cpp
    )

    target_link_libraries(ave_tests
            PRIVATE
            ave_core
    )

    # Copy test assets (WAV files) next to the test binary, if present
    add_custom_command(TARGET ave_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/assets"
            "$<TARGET_FILE_DIR:ave_tests>/assets"
    )

    # Copy FFTW DLL to test output directory
    add_custom_command(TARGET ave_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FFTW3_DIR}/libfftw3-3.dll"
            $<TARGET_FILE_DIR:ave_tests>
    )

    add_test(NAME ave_tests COMMAND ave_tests)
endif()

# --- Installation ---

install(TARGETS ave_analysis ave_core
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
        DESTINATION include/ave
)

# --- Build Info ---

message(STATUS "========================================")
message(STATUS "Audio Visual Engine - Analysis Pipeline")
message(STATUS "========================================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "FFTW3 Directory: ${FFTW3_DIR}")
message(STATUS "Build Tests: ${BUILD_TESTS}")
message(STATUS "========================================")