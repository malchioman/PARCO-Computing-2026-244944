#include <cstdio>
#include <cstdlib>
#include <random>
#include <filesystem>   // <=== aggiunto
#include "../mmio/mmio.h"

#ifdef _WIN32
    #include <direct.h>
#else
    #include <sys/stat.h>
    #include <sys/types.h>
#endif

extern "C" {
#include "../mmio/mmio.h"
}

int main(int argc, char** argv) {
    int rows = 10000, cols = 10000;
    double density = 0.5;
    int nnz = static_cast<int>(rows * cols * density);

    namespace fs = std::filesystem;


    fs::path exe_path = fs::absolute(fs::path(argv[0]));
    fs::path exe_dir  = exe_path.parent_path();

    fs::path matrices_dir = exe_dir / "matrices";
    fs::create_directories(matrices_dir);


    fs::path mtx_path = matrices_dir / "custom.mtx";

    FILE *f = fopen(mtx_path.string().c_str(), "w");
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

    std::printf("Matrice generata in %s\n", mtx_path.string().c_str());

    return 0;
}
