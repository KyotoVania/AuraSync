# 1. Image de base
FROM ubuntu:22.04

# Éviter les questions interactives
ENV DEBIAN_FRONTEND=noninteractive

# 2. Installer les dépendances :
# Ajout de 'pkg-config' nécessaire pour que CMake trouve les libs
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Dossier de travail
WORKDIR /app

# 4. Copie des sources
COPY . .

# 5. Build
RUN mkdir build && cd build && \
    cmake .. && \
    make

# 6. Commande par défaut
CMD ["./build/ave_tests"]