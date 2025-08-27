# AuraSync

## FFTW3 Installation (Windows & Linux)

This project requires [FFTW3](http://www.fftw.org/) for fast Fourier transforms. You can install it automatically using vcpkg (recommended for both Windows and Linux), or via your system package manager on Linux.

### Using vcpkg (Windows & Linux)
1. Install vcpkg: https://github.com/microsoft/vcpkg
2. Run: `vcpkg install fftw3`
3. CMake will automatically detect vcpkg if the `VCPKG_ROOT` environment variable is set.

### Linux (APT)
```sh
sudo apt update
sudo apt install libfftw3-dev
```

### Troubleshooting
If CMake fails with `FFTW3 not found`, ensure you have installed FFTW3 as above and that CMake can find it. On Windows, make sure vcpkg's toolchain file is used (set `VCPKG_ROOT`).

---

