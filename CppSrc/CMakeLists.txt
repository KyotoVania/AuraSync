cmake_minimum_required(VERSION 3.16)
project(AudioVisualEngineAnalysis VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

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

# --- Dependency: FFTW3 ---
# If using vcpkg toolchain, find FFTW3 via find_package
if(DEFINED CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg")
    message(STATUS "Using vcpkg toolchain, finding FFTW3 via find_package...")
    find_package(FFTW3 CONFIG REQUIRED)
    set(FFTW3_INCLUDE_DIRS "")
    set(FFTW3_LIBRARIES FFTW3::fftw3)
else()
    # Try to find FFTW3 using pkg-config first (for systems with it installed)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFTW3 QUIET fftw3)
    endif()

    if(NOT FFTW3_FOUND)
        # Fetch FFTW3 from source
        message(STATUS "FFTW3 not found via pkg-config, fetching from source...")
        
        FetchContent_Declare(
                fftw3
                URL https://www.fftw.org/fftw-3.3.10.tar.gz
                URL_HASH MD5=8ccbf6a5ea78a16dbc3e1306e234cc5c
        )
        
        # Configure FFTW3 build options - use only double precision
        set(ENABLE_FLOAT OFF CACHE BOOL "Build single precision library" FORCE)
        set(ENABLE_LONG_DOUBLE OFF CACHE BOOL "Build long double precision library" FORCE)
        set(BUILD_TESTS OFF CACHE BOOL "Build FFTW tests" FORCE)
        set(DISABLE_FORTRAN ON CACHE BOOL "Disable Fortran wrapper" FORCE)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
        
        FetchContent_MakeAvailable(fftw3)
        
        # Set FFTW3 variables for consistency
        set(FFTW3_INCLUDE_DIRS ${fftw3_SOURCE_DIR}/api)
        set(FFTW3_LIBRARIES fftw3)
    else()
        message(STATUS "Found FFTW3 via pkg-config")
    endif()
endif()

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
        ${FFTW3_INCLUDE_DIRS}
)

target_link_libraries(ave_core
        PUBLIC
        nlohmann_json::nlohmann_json
        ave_qmdsp
        ${FFTW3_LIBRARIES}
)


# --- Main Executable ---
add_executable(ave_analysis src/main.cpp)
target_link_libraries(ave_analysis PRIVATE ave_core)

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