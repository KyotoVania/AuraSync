# Stage 1: Builder
FROM debian:bookworm-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies needed for CMake and FFTW
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libfftw3-dev \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Copy source code (Context is already set to CppSrc by the workflow)
COPY . .

# Configure and Build
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --config Release --parallel

# Stage 2: Runtime
FROM debian:bookworm-slim

# Install runtime libraries (libfftw3-3 is required for the app to start)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfftw3-3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 runner
USER runner

# Copy the binary from the build/bin directory (as per CMakeLists.txt)
COPY --from=builder /usr/src/app/build/bin/ave_analysis /usr/local/bin/ave_analysis

# Define Entrypoint
ENTRYPOINT ["ave_analysis"]