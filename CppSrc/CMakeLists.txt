cmake_minimum_required(VERSION 3.16)
project(AudioVisualEngineAnalysis VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# --- Vcpkg setup ---
# If CMAKE_TOOLCHAIN_FILE is not set, we are not using a foreign toolchain.
# In this case, we will automatically set up vcpkg to handle our dependencies.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    # Get vcpkg by using FetchContent
    include(FetchContent)
    FetchContent_Declare(
            vcpkg
            GIT_REPOSITORY https://github.com/microsoft/vcpkg.git
    )
    FetchContent_MakeAvailable(vcpkg)

    # Set CMAKE_TOOLCHAIN_FILE to use the vcpkg toolchain.
    # We set it in the CACHE so that this block is not executed on subsequent runs.
    set(CMAKE_TOOLCHAIN_FILE "${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "Vcpkg toolchain file")
endif()
# --- End of Vcpkg setup ---


include(FetchContent)

# --- Dependency: nlohmann/json ---
FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# --- Dependency: qm-dsp ---
FetchContent_Declare(
        qmdsp
        GIT_REPOSITORY https://github.com/c4dm/qm-dsp.git
)
FetchContent_MakeAvailable(qmdsp)

# --- Dependency: FFTW ---
# With the vcpkg toolchain, we can just find the package.
# The vcpkg.json file tells vcpkg to install fftw3.
find_package(FFTW3 CONFIG REQUIRED)

# --- Library: ave_qmdsp (built from fetched qm-dsp sources) ---
add_library(ave_qmdsp STATIC
        ${qmdsp_SOURCE_DIR}/dsp/onsets/DetectionFunction.cpp
        ${qmdsp_SOURCE_DIR}/dsp/phasevocoder/PhaseVocoder.cpp
        ${qmdsp_SOURCE_DIR}/dsp/tempotracking/TempoTrackV2.cpp
        ${qmdsp_SOURCE_DIR}/dsp/transforms/FFT.cpp
        ${qmdsp_SOURCE_DIR}/ext/kissfft/kiss_fft.c
        ${qmdsp_SOURCE_DIR}/ext/kissfft/tools/kiss_fftr.c
        ${qmdsp_SOURCE_DIR}/maths/MathUtilities.cpp
        ${qmdsp_SOURCE_DIR}/maths/Correlation.cpp
        ${qmdsp_SOURCE_DIR}/dsp/signalconditioning/DFProcess.cpp
        ${qmdsp_SOURCE_DIR}/dsp/signalconditioning/Filter.cpp
        ${qmdsp_SOURCE_DIR}/dsp/signalconditioning/FiltFilt.cpp
        ${qmdsp_SOURCE_DIR}/dsp/signalconditioning/Framer.cpp
        ${qmdsp_SOURCE_DIR}/base/Pitch.cpp
        ${qmdsp_SOURCE_DIR}/dsp/chromagram/Chromagram.cpp
        ${qmdsp_SOURCE_DIR}/dsp/chromagram/ConstantQ.cpp
        ${qmdsp_SOURCE_DIR}/dsp/rateconversion/Decimator.cpp
        ${qmdsp_SOURCE_DIR}/dsp/rateconversion/DecimatorB.cpp
        ${qmdsp_SOURCE_DIR}/dsp/keydetection/GetKeyMode.cpp
)

target_include_directories(ave_qmdsp PUBLIC
        ${qmdsp_SOURCE_DIR}
        ${qmdsp_SOURCE_DIR}/include
        ${qmdsp_SOURCE_DIR}/ext/kissfft
        ${qmdsp_SOURCE_DIR}/ext/kissfft/tools
)

if (MSVC)
    target_compile_definitions(ave_qmdsp PRIVATE
        _CRT_SECURE_NO_WARNINGS
        _USE_MATH_DEFINES
    )
endif()
target_compile_definitions(ave_qmdsp PUBLIC KISS_FFT_USE_DOUBLE kiss_fft_scalar=double)


# --- Library: ave_core ---
add_library(ave_core STATIC
        src/core/AudioBuffer.cpp
        src/core/JsonContract.cpp
        src/pipeline/AnalysisPipeline.cpp
        src/pipeline/AudioLoader.cpp
        src/modules/RealBPMModule.cpp
        src/modules/RealSpectralModule.cpp
        src/modules/RealOnsetModule.cpp
        src/modules/RealTonalityModule.cpp
        src/modules/RealStructureModule.cpp
        src/modules/RealCueModule.cpp
)

target_include_directories(ave_core
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ave_core
        PUBLIC
        nlohmann_json::nlohmann_json
        ave_qmdsp
        FFTW3::fftw3
)


# --- Main Executable ---
add_executable(ave_analysis src/main.cpp)
target_link_libraries(ave_analysis PRIVATE ave_core)

# No longer need to copy DLL for fftw as vcpkg handles it

# --- Tests ---
option(BUILD_TESTS "Build tests" ON)

if(BUILD_TESTS)
    enable_testing()
    option(AVE_BUILD_ONSET_TESTS "Build granular onset tests" ON)
    set(AVE_TEST_SOURCES
            tests/test_main.cpp
            tests/test_modules.cpp
            tests/test_pipeline.cpp
            tests/test_json_contract.cpp
            tests/test_tonality_chroma.cpp
            tests/test_tonality_cmajor.cpp
            tests/test_tonality_aminor.cpp
            tests/test_structure_blocks.cpp
            tests/test_structure_constant.cpp
            tests/test_cue_synthesis.cpp
    )
    if(AVE_BUILD_ONSET_TESTS)
        list(APPEND AVE_TEST_SOURCES
                tests/test_onset_click.cpp
                tests/test_onset_silence.cpp
                tests/test_onset_ramp.cpp
                tests/test_onset_click_var_bpm.cpp
                tests/test_onset_accented.cpp
                tests/test_onset_noise_bursts.cpp
        )
    endif()

    add_executable(ave_tests ${AVE_TEST_SOURCES})
    target_link_libraries(ave_tests PRIVATE ave_core)

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/assets")
        add_custom_command(TARGET ave_tests POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/assets"
                "$<TARGET_FILE_DIR:ave_tests>/assets"
        )
    endif()

    add_test(NAME ave_tests COMMAND ave_tests)
endif()

# --- Installation ---
install(TARGETS ave_analysis ave_core
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)