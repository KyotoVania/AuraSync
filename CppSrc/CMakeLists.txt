cmake_minimum_required(VERSION 3.16)
project(AuraSyncCpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dépendance nlohmann/json (inchangé) ---
include(FetchContent)
FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# On définit notre exécutable
add_executable(AuraSyncCpp main.cpp)

# --- Configuration pour FFTW3 ---
# On attache les propriétés à la cible APRÈS sa création

# 1. On définit une variable pratique pour le chemin vers la lib FFTW3
set(FFTW3_DIR fftw-3.3.5-dll64)

# 2. On indique à CMake où trouver les fichiers .h de FFTW3
target_include_directories(AuraSyncCpp PRIVATE ${FFTW3_DIR})

# 3. On linke les librairies (nlohmann et le .lib de FFTW3 qu'on va créer)
target_link_libraries(AuraSyncCpp PRIVATE
        nlohmann_json::nlohmann_json
        ${FFTW3_DIR}/libfftw3-3.lib
)

# 4. On copie automatiquement le .dll à côté de l'exécutable après la compilation
add_custom_command(TARGET AuraSyncCpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${FFTW3_DIR}/libfftw3-3.dll"
        $<TARGET_FILE_DIR:AuraSyncCpp>
)