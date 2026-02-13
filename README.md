# AuraSync

[![Cpp_CI](https://github.com/KyotoVania/AuraSync/actions/workflows/cpp_ci.yml/badge.svg)](https://github.com/KyotoVania/AuraSync/actions/workflows/cpp_ci.yml)
[![Docker Build & Push](https://github.com/KyotoVania/AuraSync/actions/workflows/docker.yml/badge.svg)](https://github.com/KyotoVania/AuraSync/actions/workflows/docker.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://kyotovania.github.io/AuraSync/)

**AuraSync** is a high-performance C++ audio analysis engine tailored for real-time synchronization. It features BPM detection, spectral analysis, onset detection, and tonality extraction.

---

## Repository Cleanup

This repository has undergone a significant cleanup. To ensure a lightweight and focused codebase:
*   **Build artifacts**, intermediate files, and generated analysis data are now excluded via `.gitignore`.
*   **External dependencies** like `fftw3` and `qm-dsp` are no longer vendored directly in the repository. They are either installed via system package managers (`vcpkg`, `apt`, `brew`) or fetched automatically by CMake's `FetchContent` mechanism.
This approach drastically reduces repository size, improves build consistency, and provides a cleaner developer experience.

---

## Prerequisites & Dependencies

### FFTW3 Installation (Required)
This project relies on [FFTW3](http://www.fftw.org/) for fast Fourier transforms.

**Option 1: Using vcpkg (Recommended for Windows)**
1.  **Install vcpkg:** If you don't have vcpkg, clone it to a location like `C:\vcpkg` and run its bootstrap script:
    ```powershell
    cd C:\
    git clone https://github.com/microsoft/vcpkg.git
    cd vcpkg
    .\bootstrap-vcpkg.bat
    .\vcpkg integrate install
    ```
2.  **Install fftw3:**
    ```powershell
    # Ensure you are in the C:\vcpkg directory or use the full path to vcpkg.exe
    .\vcpkg install fftw3:x64-windows
    ```

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
-   CMake 3.16 or higher
-   C++ Compiler supporting C++17 (GCC, Clang, or MSVC)
-   Git (required for CMake's `FetchContent` to download dependencies like `nlohmann/json` and `qm-dsp`).

---

## Build Instructions
The source code is located in the ``CppSrc`` directory.

**Linux & macOS:**
```bash
# 1. Navigate to the project C++ source directory
cd CppSrc

# 2. Create and navigate to the build directory
mkdir build
cd build

# 3. Configure the project with CMake
cmake -DCMAKE_BUILD_TYPE=Release ..

# 4. Build the project
cmake --build . --config Release
```

**Windows (Using vcpkg):**
```powershell
# 1. Navigate to the project C++ source directory
cd CppSrc

# 2. Create and navigate to the build directory
# (Optional: remove existing build directory for a clean build)
Remove-Item -Recurse -Force build
mkdir build
cd build

# 3. Configure the project with CMake, specifying the vcpkg toolchain
#    (Replace C:/vcpkg with your actual vcpkg installation path if different)
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ..

# 4. Build the project
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

## Releases & Deployments

### Creating a Release

To trigger the **Deploy Release CI** workflow and create a new release with compiled binaries for all platforms (Linux, Windows, macOS), you need to create and push a Git tag with the format `v*` (e.g., `v1.0.0`, `v2.1.3`).

**Quick steps:**

1. **Ensure your code is ready:**
   ```bash
   # Make sure all changes are committed
   git status
   git add .
   git commit -m "Prepare for release v1.0.0"
   ```

2. **Create a tag:**
   ```bash
   # Create an annotated tag (recommended)
   git tag -a v1.0.0 -m "Release version 1.0.0"
   
   # Or create a lightweight tag
   git tag v1.0.0
   ```

3. **Push the tag to GitHub:**
   ```bash
   # Push the specific tag
   git push origin v1.0.0
   
   # Or push all tags at once
   git push --tags
   ```

4. **Monitor the deployment:**
   - Go to the [Actions tab](https://github.com/KyotoVania/AuraSync/actions) in the GitHub repository
   - Watch the **Cpp_CI** workflow run
   - The **Deploy Release** job will build binaries for all platforms and create a GitHub Release

**Important notes:**
- Tags must start with `v` to trigger the deployment (e.g., `v1.0.0`, `v2.3.1-beta`)
- The CI will automatically build for Linux, Windows, and macOS
- Release artifacts will be available in the [Releases](https://github.com/KyotoVania/AuraSync/releases) section

For more detailed information about the release process, see [RELEASE.md](RELEASE.md).

---

### Troubleshooting

**CMake Error: `fftw3` not found**
*   **Windows (vcpkg):** Ensure `vcpkg` is correctly installed and `fftw3:x64-windows` is installed using `vcpkg`. Verify that the `CMAKE_TOOLCHAIN_FILE` argument in your CMake configure command points to the correct `vcpkg.cmake` script (e.g., `-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake`).
*   **Linux/macOS:** Ensure `libfftw3-dev` (Linux) or `fftw` (macOS via Homebrew) and `pkg-config` are installed.

**Compilation Error: `'M_PI': undeclared identifier` or `kiss_fft.h` not found (Windows)**
*   This indicates a dependency issue with the `qm-dsp` library, specifically on Windows. The `CMakeLists.txt` has been configured to handle this automatically by fetching the library and applying necessary compiler definitions (`_USE_MATH_DEFINES`) and include paths. Ensure you are using the latest `CMakeLists.txt` from the repository. If the error persists, perform a clean build.

**Linker Error: `cannot open input file 'fftw3.lib'` (Windows)**
*   This suggests the linker cannot find the `fftw3` library file. The `CMakeLists.txt` has been updated to add the necessary library search paths globally. Ensure you are using the latest `CMakeLists.txt` from the repository and performing a clean build.

**CMake Error: `fatal: invalid reference: <commit_hash>`**
*   This means a `FetchContent` dependency tried to checkout an invalid Git commit hash. The `CMakeLists.txt` has been updated with a corrected commit hash. Ensure you are using the latest `CMakeLists.txt` and perform a clean build.

**CMake Error: `could not find git`**
*   Git is required for CMake `FetchContent` to download dependencies. Ensure Git is installed and in your system's PATH.