#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <iostream>

#include "mmio.h"

// mmio.c è C -> serve extern "C" in C++
extern "C" {
    int mm_read_unsymmetric_sparse(const char *fname, int *M_, int *N_, int *nz_,
                                   double **val_, int **I_, int **J_);
}

struct COOEntry {
    int32_t i, j;
    double  v;
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank=0, P=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc < 2) {
        if (rank == 0) std::cerr << "Usage: spmv_mpi <matrix.mtx>\n";
        MPI_Finalize();
        return 1;
    }

    int M=0, N=0, nz=0;
    std::vector<COOEntry> all; // only used on rank 0

    if (rank == 0) {
        int *I=nullptr, *J=nullptr;
        double *V=nullptr;

        if (mm_read_unsymmetric_sparse(argv[1], &M, &N, &nz, &V, &I, &J) != 0) {
            std::cerr << "Error reading MatrixMarket file: " << argv[1] << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        all.reserve(nz);
        for (int k=0; k<nz; k++) {
            all.push_back({(int32_t)I[k], (int32_t)J[k], V[k]});
        }

        free(I); free(J); free(V);

        std::cout << "[Rank0] Loaded: M=" << M << " N=" << N << " nz=" << nz << "\n";
    }

    // broadcast sizes so everyone knows matrix shape
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz,1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "[Info] MPI ranks=" << P << "\n";
        std::cout << "Next step: distribute COO by row owner(i)=i%P, build CSR, ghost-x exchange, SpMV.\n";
    }

    MPI_Finalize();
    return 0;
}
