# 1. Image de base (Ubuntu avec les outils de compilation)
FROM ubuntu:22.04

# Éviter les questions interactives lors de l'installation
ENV DEBIAN_FRONTEND=noninteractive

# 2. Installer les dépendances (CMake, Compilateur, FFTW, ET GIT !)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Définir le dossier de travail dans le conteneur
WORKDIR /app

# 4. Copier tout le contenu du dossier CppSrc dans le conteneur
COPY . .

# 5. Créer le dossier build, compiler le projet
RUN mkdir build && cd build && \
    cmake .. && \
    make

# 6. Commande par défaut (lancer l'analyseur ou les tests)
CMD ["./build/ave_tests"]