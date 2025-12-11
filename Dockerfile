# Stage 1: Hybrid Builder
FROM debian:bookworm-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

# Install ALL build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    pkg-config \
    libasound2-dev \
    libudev-dev \
    libx11-dev \
    libwayland-dev \
    libxkbcommon-dev \
    libfftw3-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust Toolchain (for the hybrid build part)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

WORKDIR /usr/src/app
COPY . .

# --- Build Rust (Optional side-build) ---
# We build it just in case it's a dependency or needed, but focus is C++
RUN cargo build --release

# --- Build C++ (Main Target) ---
# We use the Release build type
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --config Release --parallel

# Stage 2: Runtime (Focused on ave_analysis)
FROM debian:bookworm-slim

# Install Runtime Shared Libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libasound2 \
    libudev1 \
    libx11-6 \
    libwayland-client0 \
    libxkbcommon0 \
    libfftw3-3 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 runner
USER runner

# Copy Rust Binary (available if needed manually)
COPY --from=builder /usr/src/app/target/release/rust-visualizer /usr/local/bin/rust-visualizer

# Copy C++ Binary (The MAIN app)
# FIXED PATH: matches CMAKE_RUNTIME_OUTPUT_DIRECTORY in your CMakeLists.txt
COPY --from=builder /usr/src/app/build/bin/ave_analysis /usr/local/bin/ave_analysis

# ENTRYPOINT is strictly set to the C++ Analyzer
ENTRYPOINT ["ave_analysis"]