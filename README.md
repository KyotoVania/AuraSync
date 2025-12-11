# AuraSync

[![Cpp_CI](https://github.com/KyotoVania/AuraSync/actions/workflows/cpp_ci.yml/badge.svg)](https://github.com/KyotoVania/AuraSync/actions/workflows/cpp_ci.yml)
[![Docker Build & Push](https://github.com/KyotoVania/AuraSync/actions/workflows/docker.yml/badge.svg)](https://github.com/KyotoVania/AuraSync/actions/workflows/docker.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://kyotovania.github.io/AuraSync/)

**AuraSync** is a high-performance C++ audio analysis engine tailored for real-time synchronization. It features BPM detection, spectral analysis, onset detection, and tonality extraction.

---

## Prerequisites & Dependencies

### FFTW3 Installation (Required)
This project relies on [FFTW3](http://www.fftw.org/) for fast Fourier transforms.

**Option 1: Using vcpkg (Recommended for Windows & Linux)**
1. Install vcpkg: [Microsoft vcpkg](https://github.com/microsoft/vcpkg)
2. Run: `vcpkg install fftw3`
3. CMake will automatically detect vcpkg if the `VCPKG_ROOT` environment variable is set.

**Option 2: Linux (APT)**
```bash
sudo apt update
sudo apt install libfftw3-dev pkg-config
```
**Option 3: macOS (Homebrew)**
```bash
brew install fftw pkg-config
```

***Other Requirements***
- CMake 3.16 or higher
- C++ Compiler supporting C++17 (GCC, Clang, or MSVC)
- Git (to fetch dependencies like nlohmann/json)

---

## Build Instructions
The source code is located in the ``CppSrc`` directory.

**Linux & macOS:**
```bash
# 1. Navigate to the project directory
cd CppSrc

# 2. Create and navigate to the build directory
mkdir build && cd build

# 3. Configure the project with CMake
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

**Windows (Using vcpkg):**
```powershell
# 1. Navigate to the project directory
cd CppSrc

# 2. Create and navigate to the build directory
mkdir build; cd build

# 3. Configure the project with CMake, specifying the vcpkg toolchain
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

---

## Usage

**Running the Analysis Engine**
After building, the main executable is located in the ``bin`` directory (or ``bin/Release`` on Windows).
```bash
# From the build directory:
./bin/ave_analysis
```

**Running Tests**
To verify the integrity of the analysis modules:
```bash
# From the build directory:
# Using CTest
ctest -C Release --output-on-failure

# Or directly running the test executable
./bin/ave_tests
```

---

## Docker
You can build and run AuraSync in a containerized environment (Ubuntu-based).

**Build the Image**
Note: The build context must be the ``CppSrc`` folder.
```bash
# Run from the repository root
docker build -t aurasync -f CppSrc/Dockerfile CppSrc
```
**Run the Container**
```bash
docker run --rm -it aurasync
```

---

## Documentation
Automatic documentation is generated on every push to the ``main`` branch. (https://kyotovania.github.io/AuraSync/)

---

### Troubleshooting

**CMake Error: `FFTW3 not found`**
* Ensure `libfftw3-dev` and `pkg-config` are installed.
* On Windows, ensure `VCPKG_ROOT` is set correctly in your environment variables.

**CMake Error: `could not find git`**
* Git is required for CMake `FetchContent` to download the JSON library. Ensure Git is installed and in your PATH.