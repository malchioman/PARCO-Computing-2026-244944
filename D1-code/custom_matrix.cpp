#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <iostream>

extern "C" {
  #include "../mmio/mmio.h"
}

namespace fs = std::filesystem;

// Generate a sparse MatrixMarket "coordinate real general" matrix
// Args:
//   custom_matrix <out.mtx> <rows> <cols> <nnz_per_row>
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <out.mtx> <rows> <cols> <nnz_per_row>\n"
                  << "Example: " << argv[0] << " bin/matrices/weak_P16.mtx 65536 65536 16\n";
        return 1;
    }

    fs::path out_path = fs::path(argv[1]);
    int rows = std::max(1, std::atoi(argv[2]));
    int cols = std::max(1, std::atoi(argv[3]));
    int nnz_per_row = std::max(1, std::atoi(argv[4]));

    // Create parent dirs if needed
    if (out_path.has_parent_path()) {
        fs::create_directories(out_path.parent_path());
    }

    // Total nnz (fits int for our weak scaling choices)
    long long nnz_ll = 1LL * rows * nnz_per_row;
    if (nnz_ll > std::numeric_limits<int>::max()) {
        std::cerr << "[fatal] nnz too large for mmio int header: nnz=" << nnz_ll << "\n";
        return 2;
    }
    int nnz = (int)nnz_ll;

    FILE* f = std::fopen(out_path.string().c_str(), "w");
    if (!f) {
        std::perror("fopen");
        return 3;
    }

    // MatrixMarket header
    MM_typecode t;
    mm_initialize_typecode(&t);
    mm_set_matrix(&t);
    mm_set_coordinate(&t);
    mm_set_real(&t);
    mm_set_general(&t);

    mm_write_banner(f, t);
    mm_write_mtx_crd_size(f, rows, cols, nnz);

    // Deterministic seed (no seed to pass): depends only on (rows, cols, nnz_per_row)
    // -> Reproducibile per weak scaling senza chiedere input extra.
    uint64_t seed = 1469598103934665603ULL;
    seed ^= (uint64_t)rows;       seed *= 1099511628211ULL;
    seed ^= (uint64_t)cols;       seed *= 1099511628211ULL;
    seed ^= (uint64_t)nnz_per_row;seed *= 1099511628211ULL;

    std::mt19937 rng((uint32_t)(seed ^ (seed >> 32)));

    std::uniform_int_distribution<int> Cj(1, cols);
    std::uniform_real_distribution<double> Val(-10.0, 10.0);

    // For each row, choose nnz_per_row unique columns
    std::vector<int> cols_row;
    cols_row.reserve((size_t)nnz_per_row);

    for (int i = 1; i <= rows; ++i) {
        cols_row.clear();
        while ((int)cols_row.size() < nnz_per_row) {
            int c = Cj(rng);
            cols_row.push_back(c);
            std::sort(cols_row.begin(), cols_row.end());
            cols_row.erase(std::unique(cols_row.begin(), cols_row.end()), cols_row.end());
        }

        // Write row entries
        for (int k = 0; k < nnz_per_row; ++k) {
            double v = Val(rng);
            std::fprintf(f, "%d %d %.6f\n", i, cols_row[k], v);
        }
    }

    std::fclose(f);

    std::cout << "Matrix generated: " << out_path.string()
              << " (" << rows << "x" << cols
              << ", nnz=" << nnz
              << ", nnz_per_row=" << nnz_per_row << ")\n";
    return 0;
}
