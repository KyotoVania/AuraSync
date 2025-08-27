#include <iostream>
#include <cmath> // Pour la fonction cos()
#include <fftw3.h>
#include <vector>
//Mathématiques

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// On définit une taille pour notre signal
#define N 8

int main() {
    std::cout << "--- Test de la librairie FFTW3 ---" << std::endl;

    // 1. Allouer la mémoire pour l'entrée et la sortie
    // On utilise fftw_malloc pour une meilleure performance (alignement mémoire)
    double* in;
    fftw_complex* out;
    in = (double*) fftw_malloc(sizeof(double) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));

    // 2. Créer un "plan" FFTW
    // C'est l'étape où FFTW optimise le calcul pour une taille N donnée.
    // On fait une transformée "réel vers complexe" (r2c) en 1 dimension (1d).
    fftw_plan p;
    p = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);

    // 3. Remplir le tableau d'entrée avec un signal simple
    // Ici, une onde cosinus qui fait un tour complet sur les N échantillons.
    std::cout << "\nSignal d'entrée (cosinus) :" << std::endl;
    for (int i = 0; i < N; i++) {
        in[i] = cos(2 * M_PI * i / N);
        std::cout << "in[" << i << "] = " << in[i] << std::endl;
    }

    // 4. Exécuter le plan pour faire le calcul
    fftw_execute(p);

    // 5. Afficher les résultats
    // La sortie est un tableau de nombres complexes (partie réelle, partie imaginaire)
    std::cout << "\nSignal de sortie (transformée de Fourier) :" << std::endl;
    for (int i = 0; i < (N / 2 + 1); i++) {
        double real = out[i][0]; // Partie réelle
        double imag = out[i][1]; // Partie imaginaire
        std::cout << "out[" << i << "] = " << real << " + " << imag << "i" << std::endl;
    }

    // 6. Nettoyer
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    std::cout << "\n--- Test terminé avec succès ! ---" << std::endl;

    return 0;
}