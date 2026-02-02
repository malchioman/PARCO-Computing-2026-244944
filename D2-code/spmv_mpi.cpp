#include <mpi.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>

#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>

#include "mmio.h"

// mmio.c è C -> serve extern "C" in C++
extern "C" {
int mm_read_unsymmetric_sparse(const char *fname, int *M_, int *N_, int *nz_,
                               double **val_, int **I_, int **J_);
}

namespace fs = std::filesystem;

struct COOEntry {
    int32_t i;   // global row (0-based)
    int32_t j;   // global col (0-based)
    double  v;
};

static inline int local_row_of_global(int global_i, int rank, int P) {
    // valido solo se global_i % P == rank
    return (global_i - rank) / P;
}

static inline int num_local_rows_cyclic(int M, int rank, int P) {
    if (rank >= M) return 0;
    return 1 + (M - 1 - rank) / P;
}

// ---------------------- CSR local ----------------------
static void coo_to_csr_cyclic_rows(const std::vector<COOEntry>& coo_local,
                                  int M_global, int N_global, int rank, int P,
                                  std::vector<int>& rowptr,
                                  std::vector<int>& colind,
                                  std::vector<double>& vals)
{
    (void)N_global;

    const int localM = num_local_rows_cyclic(M_global, rank, P);
    rowptr.assign(localM + 1, 0);

    // Count nnz per local row
    for (const auto& e : coo_local) {
        if (e.i < 0 || e.i >= M_global) continue;
        if (e.i % P != rank) continue;
        int lr = local_row_of_global((int)e.i, rank, P);
        if (lr >= 0 && lr < localM) rowptr[lr + 1]++;
    }

    // Prefix sum -> rowptr
    for (int r = 0; r < localM; r++) rowptr[r + 1] += rowptr[r];

    const int local_nnz = rowptr[localM];
    colind.assign(local_nnz, 0);
    vals.assign(local_nnz, 0.0);

    // Temp cursor = copy of rowptr
    std::vector<int> cursor = rowptr;

    // Fill
    for (const auto& e : coo_local) {
        if (e.i % P != rank) continue;
        int lr = local_row_of_global((int)e.i, rank, P);
        if (lr < 0 || lr >= localM) continue;
        int pos = cursor[lr]++;
        colind[pos] = (int)e.j;
        vals[pos]   = e.v;
    }
}

static void spmv_csr_local(const std::vector<int>& rowptr,
                           const std::vector<int>& colind,
                           const std::vector<double>& vals,
                           const std::vector<double>& x_global,
                           std::vector<double>& y_local)
{
    const int localM = (int)rowptr.size() - 1;
    y_local.assign(localM, 0.0);

    for (int r = 0; r < localM; r++) {
        double sum = 0.0;
        for (int k = rowptr[r]; k < rowptr[r + 1]; k++) {
            sum += vals[k] * x_global[colind[k]];
        }
        y_local[r] = sum;
    }
}

// ---------------------- Serial validation (rank0) ----------------------
struct CSRd {
    int nrows=0, ncols=0;
    std::vector<int> rowptr;
    std::vector<int> col;
    std::vector<double> val;
};

static CSRd build_csr_serial_from_mmio_unsym(const char* path) {
    int M=0, N=0, nz=0;
    int *I=nullptr, *J=nullptr;
    double *V=nullptr;

    if (mm_read_unsymmetric_sparse(path, &M, &N, &nz, &V, &I, &J) != 0) {
        std::cerr << "[fatal] mmio read failed for " << path << "\n";
        std::exit(1);
    }

    CSRd A;
    A.nrows = M;
    A.ncols = N;
    A.rowptr.assign(M+1, 0);

    for (int k=0;k<nz;k++) A.rowptr[I[k]+1]++;

    for (int r=0;r<M;r++) A.rowptr[r+1] += A.rowptr[r];

    A.col.assign(nz, 0);
    A.val.assign(nz, 0.0);
    std::vector<int> next = A.rowptr;

    for (int k=0;k<nz;k++){
        int r = I[k];
        int p = next[r]++;
        A.col[p] = J[k];
        A.val[p] = V[k];
    }

    free(I); free(J); free(V);
    return A;
}

static std::vector<double> spmv_csr_serial_double(const CSRd& A, const std::vector<double>& x) {
    std::vector<double> y(A.nrows, 0.0);
    for (int i=0;i<A.nrows;i++){
        double acc=0.0;
        for (int k=A.rowptr[i]; k<A.rowptr[i+1]; k++){
            acc += A.val[k] * x[A.col[k]];
        }
        y[i]=acc;
    }
    return y;
}

static void compute_validation(const std::vector<double>& y,
                               const std::vector<double>& yref,
                               double& rel_L2,
                               double& max_abs)
{
    long double diff2=0.0L, ref2=0.0L;
    long double maxa=0.0L;

    const int n = (int)std::min(y.size(), yref.size());
    for (int i=0;i<n;i++){
        long double d = (long double)y[i] - (long double)yref[i];
        diff2 += d*d;
        ref2  += (long double)yref[i]*(long double)yref[i];
        long double ad = fabsl(d);
        if (ad>maxa) maxa=ad;
    }
    long double rel = (ref2>0.0L) ? sqrtl(diff2/ref2) : sqrtl(diff2);

    rel_L2 = (double)rel;
    max_abs = (double)maxa;
}

// ---------------------- Logging (rank0 only) ----------------------
static void append_results_rank0(const std::string& matrix_path,
                                 int M, int N, int nz,
                                 int P, int iters,
                                 double spmv_time_s,     // avg time/iter (max ranks)
                                 double gflops,
                                 bool   has_validation,
                                 double rel_L2,
                                 double max_abs)
{
    try {
        // ✅ Scrive SEMPRE nel root del repo anche se lanci da bin/
        // eseguibile è in bin/ -> results è in ../results
        fs::path results_dir = fs::path("..") / "results";
        fs::create_directories(results_dir);

        fs::path log_path = results_dir / "spmv_mpi_results.txt";
        bool write_header = !fs::exists(log_path);

        std::ofstream fout(log_path, std::ios::app);
        if (!fout) {
            std::cerr << "[warning] Could not write results to: " << log_path << "\n";
            return;
        }

        if (write_header) {
            fout << "SpMV MPI Benchmark - Run Log\n\n";
        }

        // timestamp
        auto now = std::chrono::system_clock::now();
        std::time_t t_c = std::chrono::system_clock::to_time_t(now);
        std::tm* tm = std::localtime(&t_c);

        fout << "=================================================================\n";
        fout << "Run at " << std::put_time(tm, "%Y-%m-%d %H:%M:%S") << "\n";
        fout << "Matrix    : " << matrix_path << " (" << M << " x " << N << ", nz=" << nz << ")\n";
        fout << "MPI       : ranks=" << P << ", iters=" << iters << "\n";
        fout << std::fixed << std::setprecision(6);
        fout << "Timing    : avg_spmv_time_s=" << spmv_time_s << "\n";
        fout << std::fixed << std::setprecision(3);
        fout << "Perf      : GFLOPS=" << gflops << "\n";

        if (has_validation) {
            fout.setf(std::ios::scientific, std::ios::floatfield);
            fout << "Validation: rel_L2_error=" << rel_L2
                 << ", max_abs_error=" << max_abs << "\n";
            fout.unsetf(std::ios::floatfield);
        } else {
            fout << "Validation: (not computed)\n";
        }

        fout << "=================================================================\n\n";

    } catch (const std::exception& e) {
        std::cerr << "[warning] Exception while writing results file: " << e.what() << "\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, P = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: spmv_mpi <matrix.mtx> [iters]\n";
            std::cerr << "Note: this baseline supports MatrixMarket 'real sparse general' files.\n";
        }
        MPI_Finalize();
        return 1;
    }

    const char* path = argv[1];
    int iters = 10;
    if (argc >= 3) iters = std::max(1, std::atoi(argv[2]));

    int M = 0, N = 0, nz = 0;
    std::vector<COOEntry> coo_local;

    // Create MPI datatype for COOEntry
    MPI_Datatype MPI_COO;
    {
        COOEntry dummy{};
        int blocklen[3] = {1, 1, 1};
        MPI_Aint disp[3];
        MPI_Aint base;
        MPI_Get_address(&dummy, &base);
        MPI_Get_address(&dummy.i, &disp[0]);
        MPI_Get_address(&dummy.j, &disp[1]);
        MPI_Get_address(&dummy.v, &disp[2]);
        for (int t = 0; t < 3; t++) disp[t] -= base;

        MPI_Datatype types[3] = {MPI_INT32_T, MPI_INT32_T, MPI_DOUBLE};
        MPI_Type_create_struct(3, blocklen, disp, types, &MPI_COO);
        MPI_Type_commit(&MPI_COO);
    }

    // -------- Rank 0 reads and distributes COO by row owner(i)=i%P --------
    if (rank == 0) {
        int *I = nullptr, *J = nullptr;
        double *V = nullptr;

        if (mm_read_unsymmetric_sparse(path, &M, &N, &nz, &V, &I, &J) != 0) {
            std::cerr << "Error reading MatrixMarket file: " << path << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::vector< std::vector<COOEntry> > buckets(P);

        for (int k = 0; k < nz; k++) {
            int gi = I[k];
            int gj = J[k];
            double val = V[k];
            int owner = gi % P;
            buckets[owner].push_back({(int32_t)gi, (int32_t)gj, val});
        }

        free(I); free(J); free(V);

        for (int p = 1; p < P; p++) {
            int cnt = (int)buckets[p].size();
            MPI_Send(&cnt, 1, MPI_INT, p, 100, MPI_COMM_WORLD);
            if (cnt > 0) {
                MPI_Send(buckets[p].data(), cnt, MPI_COO, p, 101, MPI_COMM_WORLD);
            }
        }

        coo_local = std::move(buckets[0]);

        std::cout << "[Rank0] Loaded: M=" << M << " N=" << N << " nz=" << nz << "\n";
        std::cout << "[Rank0] MPI ranks=" << P << " iters=" << iters << "\n";
    } else {
        int cnt = 0;
        MPI_Recv(&cnt, 1, MPI_INT, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        coo_local.resize(cnt);
        if (cnt > 0) {
            MPI_Recv(coo_local.data(), cnt, MPI_COO, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Broadcast sizes
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // -------- Build local CSR --------
    std::vector<int> rowptr, colind;
    std::vector<double> vals;
    coo_to_csr_cyclic_rows(coo_local, M, N, rank, P, rowptr, colind, vals);

    // -------- Build x (baseline: x=1) and broadcast --------
    std::vector<double> x(N, 1.0);
    MPI_Bcast(x.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // -------- SpMV iterations & timing (measure only SpMV compute) --------
    std::vector<double> y_local;
    MPI_Barrier(MPI_COMM_WORLD);

    double t0 = MPI_Wtime();
    for (int it = 0; it < iters; it++) {
        spmv_csr_local(rowptr, colind, vals, x, y_local);
    }
    double t1 = MPI_Wtime();

    double local_time = (t1 - t0) / iters;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // -------- Gather results to rank0 as (global_row, y_val) --------
    const int localM = (int)y_local.size();
    std::vector<int> global_rows(localM);
    for (int lr = 0; lr < localM; lr++) {
        global_rows[lr] = rank + lr * P;
    }

    std::vector<int> recv_counts, displs;
    if (rank == 0) recv_counts.resize(P, 0);

    MPI_Gather(&localM, 1, MPI_INT,
               rank == 0 ? recv_counts.data() : nullptr, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    int total_rows = 0;
    if (rank == 0) {
        displs.resize(P, 0);
        for (int p = 0; p < P; p++) {
            displs[p] = total_rows;
            total_rows += recv_counts[p];
        }
    }

    std::vector<int> all_grows;
    std::vector<double> all_yvals;
    if (rank == 0) {
        all_grows.resize(total_rows);
        all_yvals.resize(total_rows);
    }

    MPI_Gatherv(global_rows.data(), localM, MPI_INT,
                rank == 0 ? all_grows.data() : nullptr,
                rank == 0 ? recv_counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Gatherv(y_local.data(), localM, MPI_DOUBLE,
                rank == 0 ? all_yvals.data() : nullptr,
                rank == 0 ? recv_counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // -------- Rank0: rebuild y, validate, log --------
    if (rank == 0) {
        std::vector<double> y(M, 0.0);
        for (int idx = 0; idx < total_rows; idx++) {
            int gi = all_grows[idx];
            if (0 <= gi && gi < M) y[gi] = all_yvals[idx];
        }

        double gflops = (max_time > 0.0) ? (2.0 * (double)nz) / max_time / 1e9 : 0.0;

        bool has_validation = false;
        double rel_L2 = 0.0, max_abs = 0.0;

        try {
            CSRd Aref = build_csr_serial_from_mmio_unsym(path);
            if (Aref.nrows == M && Aref.ncols == N) {
                std::vector<double> yref = spmv_csr_serial_double(Aref, x);
                compute_validation(y, yref, rel_L2, max_abs);
                has_validation = true;

                std::cerr.setf(std::ios::scientific, std::ios::floatfield);
                std::cerr << "[validation] rel_L2_error=" << rel_L2
                          << " max_abs_error=" << max_abs << "\n";
                std::cerr.unsetf(std::ios::floatfield);
            } else {
                std::cerr << "[validation] shape mismatch, skipped.\n";
            }
        } catch (...) {
            std::cerr << "[validation] exception, skipped.\n";
        }

        std::cout << "[Rank0] Avg SpMV time/iter (max over ranks): " << max_time << " s\n";
        std::cout << "[Rank0] GFLOPS: " << gflops << "\n";
        std::cout << "[Rank0] y[0]=" << (M > 0 ? y[0] : 0.0)
                  << "  y[last]=" << (M > 0 ? y[M-1] : 0.0) << "\n";

        append_results_rank0(path, M, N, nz, P, iters, max_time, gflops,
                             has_validation, rel_L2, max_abs);

        std::cout << "[Rank0] Results appended to ../results/spmv_mpi_results.txt\n";
    }

    MPI_Type_free(&MPI_COO);
    MPI_Finalize();
    return 0;
}
