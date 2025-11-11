#include <cstdio>
#include <cstdlib>
#include <random>
#include "mmio.h"

#ifdef _WIN32
    #include <direct.h>
#else
    #include <sys/stat.h>
    #include <sys/types.h>
#endif

int main() {
    int rows = 10000, cols = 10000;
    double density = 0.5;
    int nnz = static_cast<int>(rows * cols * density);

#ifdef _WIN32
    _mkdir("matrices");
#else
    mkdir("matrices", 0755);
#endif

    FILE *f = fopen("matrices/large_random.mtx", "w");
    if (!f) {
        perror("Errore apertura file");
        return 1;
    }

    MM_typecode t;
    mm_initialize_typecode(&t);
    mm_set_matrix(&t);
    mm_set_coordinate(&t);
    mm_set_real(&t);
    mm_set_general(&t);

    mm_write_banner(f, t);
    mm_write_mtx_crd_size(f, rows, cols, nnz);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> Ri(1, rows), Cj(1, cols);
    std::uniform_real_distribution<double> Val(-10.0, 10.0);

    for (int k = 0; k < nnz; ++k)
        fprintf(f, "%d %d %.6f\n", Ri(rng), Cj(rng), Val(rng));

    fclose(f);
    printf("Matrice generata in matrices/large_random.mtx\n");
    return 0;
}
