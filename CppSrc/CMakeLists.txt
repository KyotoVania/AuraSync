cmake_minimum_required(VERSION 3.16)
project(AudioVisualEngineAnalysis VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include(FetchContent)
FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

if(WIN32)
    set(FFTW3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/fftw-3.3.5-dll64)
    include_directories(${FFTW3_DIR})
    set(FFTW_LIBRARIES ${FFTW3_DIR}/libfftw3-3.lib)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFTW3 REQUIRED fftw3)
    include_directories(${FFTW3_INCLUDE_DIRS})
    link_directories(${FFTW3_LIBRARY_DIRS}) # FIX: Added for Mac/Linux linking
    set(FFTW_LIBRARIES ${FFTW3_LIBRARIES})
endif()

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
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/qm-dsp
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/qm-dsp/include
)

add_library(ave_qmdsp STATIC
        lib/qm-dsp/dsp/onsets/DetectionFunction.cpp
        lib/qm-dsp/dsp/phasevocoder/PhaseVocoder.cpp
        lib/qm-dsp/dsp/tempotracking/TempoTrackV2.cpp
        lib/qm-dsp/dsp/transforms/FFT.cpp
        lib/qm-dsp/ext/kissfft/kiss_fft.c
        lib/qm-dsp/ext/kissfft/tools/kiss_fftr.c
        lib/qm-dsp/maths/MathUtilities.cpp
        lib/qm-dsp/maths/Correlation.cpp
        lib/qm-dsp/dsp/signalconditioning/DFProcess.cpp
        lib/qm-dsp/dsp/signalconditioning/Filter.cpp
        lib/qm-dsp/dsp/signalconditioning/FiltFilt.cpp
        lib/qm-dsp/dsp/signalconditioning/Framer.cpp
        lib/qm-dsp/base/Pitch.cpp
        lib/qm-dsp/dsp/chromagram/Chromagram.cpp
        lib/qm-dsp/dsp/chromagram/ConstantQ.cpp
        lib/qm-dsp/dsp/rateconversion/Decimator.cpp
        lib/qm-dsp/dsp/rateconversion/DecimatorB.cpp
        lib/qm-dsp/dsp/keydetection/GetKeyMode.cpp
)

target_include_directories(ave_qmdsp PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/qm-dsp
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/qm-dsp/include
)

if (MSVC)
    target_compile_definitions(ave_qmdsp PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
target_compile_definitions(ave_qmdsp PUBLIC KISS_FFT_USE_DOUBLE kiss_fft_scalar=double)

target_link_libraries(ave_core
        PUBLIC
        nlohmann_json::nlohmann_json
        ave_qmdsp
        ${FFTW_LIBRARIES}
)

add_executable(ave_analysis src/main.cpp)
target_link_libraries(ave_analysis PRIVATE ave_core)

if(WIN32)
    add_custom_command(TARGET ave_analysis POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FFTW3_DIR}/libfftw3-3.dll"
            $<TARGET_FILE_DIR:ave_analysis>
    )
endif()

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

    if(WIN32)
        add_custom_command(TARGET ave_tests POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FFTW3_DIR}/libfftw3-3.dll"
                $<TARGET_FILE_DIR:ave_tests>
        )
    endif()

    add_test(NAME ave_tests COMMAND ave_tests)
endif()

install(TARGETS ave_analysis ave_core
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)