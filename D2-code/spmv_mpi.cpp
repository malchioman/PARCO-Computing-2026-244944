#include <mpi.h>
#include <omp.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cctype>
#include <ctime>

#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <limits>

namespace fs = std::filesystem;

// ============================================================
// Debug / safety
// ============================================================

#ifndef SPMV_CHECK_G2L
  #ifdef NDEBUG
    #define SPMV_CHECK_G2L 0
  #else
    #define SPMV_CHECK_G2L 1
  #endif
#endif

// ============================================================
// Types
// ============================================================

struct COOEntry {
    int32_t i;   // 0-based row
    int32_t j;   // 0-based col
    double  v;
};

struct CSRLocal {
    int localM = 0;
    std::vector<int> rowptr;
    std::vector<int> col;
    std::vector<double> val;
};

struct ValidationResult {
    long double rel_L2_error = 0.0L;
    long double max_abs_error = 0.0L;
};

// ============================================================
// Small utilities (D1-like)
// ============================================================

static std::string lower_copy(std::string s){
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

static bool file_exists(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (f) { std::fclose(f); return true; }
    return false;
}

static std::string rtrim_cr(const std::string& s) {
    if (!s.empty() && s.back() == '\r') return s.substr(0, s.size()-1);
    return s;
}

// Find matrix path like D1 (but only rank0 does it), then broadcast to all ranks.
static std::string resolve_matrix_path_rank0(const std::string& mtxArg) {
    std::string mtx = mtxArg;

    if (file_exists(mtx)) return mtx;

    const char* baseEnv = std::getenv("MATRICES_DIR");
    if (baseEnv) {
        std::string candidate = std::string(baseEnv) + "/" + mtxArg;
        if (file_exists(candidate)) return candidate;
    }

    {
        std::string candidate = "matrices/" + mtxArg;
        if (file_exists(candidate)) return candidate;
    }
    {
        std::string candidate = "bin/matrices/" + mtxArg;
        if (file_exists(candidate)) return candidate;
    }
    {
        std::string candidate = "../bin/matrices/" + mtxArg;
        if (file_exists(candidate)) return candidate;
    }

    return ""; // not found
}

static std::string bcast_string_from_rank0(std::string s, int rank) {
    int len = (rank == 0) ? (int)s.size() : 0;
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) s.assign((size_t)len, '\0');
    MPI_Bcast(s.data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
    return s;
}

static fs::path repo_root_from_exe(const char* argv0) {
    // come D1: exe in <repo>/bin/spmv_mpi -> root = parent(bin)
    try {
        fs::path exe = fs::canonical(fs::path(argv0));
        fs::path bin = exe.parent_path();
        return bin.parent_path();
    } catch (...) {
        // fallback
        return fs::current_path().parent_path();
    }
}

static int omp_used_threads()
{
    int used = 1;
    #pragma omp parallel
    {
        #pragma omp master
        used = omp_get_num_threads();
    }
    return used;
}

// ============================================================
// Deterministic "random-like" x[j] (no need to store full x)
// (Gaussian mean 0 std 10, seed fixed)
// ============================================================

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline double u01_from_u64(uint64_t x) {
    // 53-bit mantissa uniform in (0,1)
    const double inv = 1.0 / 9007199254740992.0; // 2^53
    double u = ((x >> 11) & 0x1fffffffffffffULL) * inv;
    // avoid 0
    if (u <= 0.0) u = inv;
    if (u >= 1.0) u = 1.0 - inv;
    return u;
}

static inline double x_value(int j) {
    // Box-Muller using deterministic hashed uniforms
    const uint64_t seed = 42ULL;
    uint64_t a = splitmix64(seed ^ (uint64_t)j);
    uint64_t b = splitmix64(a + 0x12345678ULL);

    double u1 = u01_from_u64(a);
    double u2 = u01_from_u64(b);

    const double TWO_PI = 2.0 * std::acos(-1.0);

    double R = std::sqrt(-2.0 * std::log(u1));
    double T = TWO_PI * u2;
    double z = R * std::cos(T); // N(0,1)
    return 10.0 * z;            // N(0,10)
}

// ============================================================
// Partition helpers
// ============================================================

static inline int owner_row(int i, int P) { return i % P; }

static inline int local_row_of_global(int gi, int rank, int P) {
    // valid only if gi%P==rank
    return (gi - rank) / P;
}

static inline int num_local_rows_cyclic(int M, int rank, int P) {
    if (rank >= M) return 0;
    return 1 + (M - 1 - rank) / P;
}

// ============================================================
// MPI datatype for COOEntry (for Gatherv validation)
// ============================================================

static MPI_Datatype make_mpi_coo_type()
{
    MPI_Datatype MPI_COO;
    COOEntry d{};
    int bl[3] = {1,1,1};
    MPI_Aint disp[3], base;
    MPI_Get_address(&d, &base);
    MPI_Get_address(&d.i, &disp[0]);
    MPI_Get_address(&d.j, &disp[1]);
    MPI_Get_address(&d.v, &disp[2]);
    for (int t=0;t<3;t++) disp[t] -= base;
    MPI_Datatype types[3] = {MPI_INT32_T, MPI_INT32_T, MPI_DOUBLE};
    MPI_Type_create_struct(3, bl, disp, types, &MPI_COO);
    MPI_Type_commit(&MPI_COO);
    return MPI_COO;
}

// ============================================================
// MPI-IO helpers: read large ranges in pieces (avoid int limit)
// ============================================================

static void mpi_file_read_at_all_big(MPI_File fh, MPI_Offset off,
                                     char* buf, MPI_Offset len)
{
    const MPI_Offset MAX = (MPI_Offset)std::numeric_limits<int>::max();
    MPI_Offset done = 0;
    while (done < len) {
        MPI_Offset chunk = std::min(MAX, len - done);
        MPI_Status st;
        MPI_File_read_at_all(fh, off + done, buf + (size_t)done, (int)chunk, MPI_CHAR, &st);
        done += chunk;
    }
}

// ============================================================
// MatrixMarket parsing (MPI-IO, "coordinate real" robust)
// ============================================================

static bool parse_dims_line(const std::string& line, int& M, int& N, int& nz) {
    std::istringstream iss(line);
    long long m,n,nnz;
    if (!(iss >> m >> n >> nnz)) return false;
    if (m <= 0 || n <= 0 || nnz < 0) return false;
    M = (int)m;
    N = (int)n;
    nz = (int)nnz;
    return true;
}

static bool parse_triplet_line(const std::string& line, int& i0, int& j0, double& v) {
    std::istringstream iss(line);
    long long i, j;
    double val;
    if (!(iss >> i >> j >> val)) return false;
    i0 = (int)i - 1;
    j0 = (int)j - 1;
    v  = val;
    return true;
}

// BONUS 4: parallel matrix reading with MPI-IO chunk parsing
static void parallel_read_matrix_market_mpiio(const char* path,
                                              int rank, int P,
                                              int& M, int& N, int& nz_header,
                                              std::vector<COOEntry>& coo_local)
{
    MPI_File fh;
    int rc = MPI_File_open(MPI_COMM_WORLD, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (rc != MPI_SUCCESS) {
        if (rank == 0) std::cerr << "[fatal] MPI_File_open failed for: " << path << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Offset fsize = 0;
    MPI_File_get_size(fh, &fsize);

    MPI_Offset data_start = 0;
    int M0=0, N0=0, nz0=0;

    if (rank == 0) {
        const MPI_Offset HEAD_MAX = std::min<MPI_Offset>(fsize, (MPI_Offset)(8LL<<20)); // 8MB
        std::vector<char> head((size_t)HEAD_MAX, '\0');

        MPI_Status st;
        MPI_File_read_at(fh, 0, head.data(), (int)HEAD_MAX, MPI_CHAR, &st);

        // Banner check (soft)
        {
            size_t eol = 0;
            while (eol < head.size() && head[eol] != '\n') eol++;
            std::string banner(head.data(), head.data() + eol);
            banner = rtrim_cr(banner);
            if (banner.rfind("%%MatrixMarket", 0) != 0) {
                std::cerr << "[warning] File does not start with %%MatrixMarket banner\n";
            } else {
                if (banner.find("matrix") == std::string::npos ||
                    banner.find("coordinate") == std::string::npos) {
                    std::cerr << "[warning] MatrixMarket banner not 'matrix coordinate ...'\n";
                }
                if (banner.find("real") == std::string::npos) {
                    std::cerr << "[warning] MatrixMarket data type not 'real' (parser expects i j val)\n";
                }
            }
        }

        // scan lines and find dims line using real '\n' offsets
        size_t pos = 0;
        bool dims_found = false;

        while (pos < head.size()) {
            size_t eol = pos;
            while (eol < head.size() && head[eol] != '\n') eol++;

            std::string line(head.data() + pos, head.data() + eol);
            line = rtrim_cr(line);

            size_t next_pos = (eol < head.size() && head[eol] == '\n') ? (eol + 1) : eol;

            if (!line.empty() && line[0] != '%') {
                if (parse_dims_line(line, M0, N0, nz0)) {
                    data_start = (MPI_Offset)next_pos;
                    dims_found = true;
                    break;
                }
            }
            pos = next_pos;
        }

        if (!dims_found) {
            std::cerr << "[fatal] Could not parse dims line in header chunk (increase HEAD_MAX)\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&M0, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N0, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz0, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast data_start via long long for portability
    long long tmp = 0;
    if (rank == 0) tmp = (long long)data_start;
    MPI_Bcast(&tmp, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    data_start = (MPI_Offset)tmp;

    M = M0; N = N0; nz_header = nz0;

    const MPI_Offset data_bytes = fsize - data_start;
    const MPI_Offset chunk = (data_bytes + P - 1) / P;

    MPI_Offset my_start = data_start + (MPI_Offset)rank * chunk;
    MPI_Offset my_end   = std::min<MPI_Offset>(data_start + (MPI_Offset)(rank + 1) * chunk, fsize);

    // overlap to handle boundary lines
    const MPI_Offset OVER = 4096;
    MPI_Offset read_start = (rank == 0) ? my_start : std::max<MPI_Offset>(data_start, my_start - OVER);
    MPI_Offset read_end   = (rank == P - 1) ? my_end   : std::min<MPI_Offset>(fsize, my_end + OVER);

    MPI_Offset read_len = read_end - read_start;

    std::vector<char> raw((size_t)read_len, '\0');
    mpi_file_read_at_all_big(fh, read_start, raw.data(), read_len);

    MPI_File_close(&fh);

    std::string s(raw.data(), raw.size());

    // Convert absolute [my_start, my_end) to indices relative to read_start
    MPI_Offset rel_my_start = my_start - read_start;
    MPI_Offset rel_my_end   = my_end   - read_start;

    size_t a = (size_t)std::max<MPI_Offset>(0, rel_my_start);
    size_t b = (size_t)std::max<MPI_Offset>(0, rel_my_end);
    if (a > s.size()) a = s.size();
    if (b > s.size()) b = s.size();

    // Align start/end to line boundaries
    if (rank != 0) {
        while (a < s.size() && s[a] != '\n') a++;
        if (a < s.size()) a++;
    }
    if (rank != P - 1) {
        while (b > 0 && s[b - 1] != '\n') b--;
    }

    coo_local.clear();
    if (b <= a) return;

    std::istringstream iss(s.substr(a, b - a));
    std::string line;

    coo_local.reserve((size_t)std::max(1, nz_header / P));

    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        line = rtrim_cr(line);
        if (line.empty()) continue;
        if (!line.empty() && line[0] == '%') continue;

        int i0, j0;
        double v;
        if (!parse_triplet_line(line, i0, j0, v)) continue;
        if (i0 < 0 || i0 >= M || j0 < 0 || j0 >= N) continue;

        // Keep only rows owned by this rank (cyclic row partition)
        if (owner_row(i0, P) == rank) {
            coo_local.push_back({(int32_t)i0, (int32_t)j0, v});
        }
    }
}

// ============================================================
// COO -> CSR local
// ============================================================

static CSRLocal coo_to_csr_cyclic_rows(const std::vector<COOEntry>& coo_local,
                                       int M_global, int rank, int P)
{
    CSRLocal A;
    A.localM = num_local_rows_cyclic(M_global, rank, P);
    A.rowptr.assign(A.localM + 1, 0);

    for (const auto& e : coo_local) {
        int lr = local_row_of_global((int)e.i, rank, P);
        if (0 <= lr && lr < A.localM) A.rowptr[lr + 1]++;
    }

    for (int r = 0; r < A.localM; ++r)
        A.rowptr[r + 1] += A.rowptr[r];

    const int local_nnz = A.rowptr[A.localM];
    A.col.assign(local_nnz, 0);
    A.val.assign(local_nnz, 0.0);

    std::vector<int> cursor = A.rowptr;
    for (const auto& e : coo_local) {
        int lr = local_row_of_global((int)e.i, rank, P);
        if (lr < 0 || lr >= A.localM) continue;
        int pos = cursor[lr]++;
        A.col[pos] = (int)e.j;
        A.val[pos] = e.v;
    }

    return A;
}

// Optional (wow) sort by column within each CSR row
static void sort_csr_rows_by_col(CSRLocal& A)
{
    for (int r = 0; r < A.localM; ++r) {
        int a = A.rowptr[r];
        int b = A.rowptr[r + 1];
        int len = b - a;
        if (len <= 1) continue;

        std::vector<int> idx(len);
        for (int k = 0; k < len; ++k) idx[k] = a + k;

        std::sort(idx.begin(), idx.end(),
                  [&](int p, int q){ return A.col[p] < A.col[q]; });

        std::vector<int>    ctmp(len);
        std::vector<double> vtmp(len);
        for (int k = 0; k < len; ++k) {
            ctmp[k] = A.col[idx[k]];
            vtmp[k] = A.val[idx[k]];
        }
        for (int k = 0; k < len; ++k) {
            A.col[a + k] = ctmp[k];
            A.val[a + k] = vtmp[k];
        }
    }
}

// ============================================================
// BONUS 1+2: Build x_local (owned + ghosts) via Alltoallv request/response
// g2l: size N, g2l[col] -> index in x_local (owned or ghost)
// ============================================================

static void build_x_and_exchange_ghosts_alltoallv(int rank, int P, int N,
                                                  const std::vector<int>& colind,
                                                  std::vector<double>& x_local,
                                                  std::vector<int>& g2l)
{
    g2l.assign(N, -1);
    x_local.clear();

    // Owned x entries (cyclic by column owner = j%P)
    for (int j = rank; j < N; j += P) {
        g2l[j] = (int)x_local.size();
        x_local.push_back(x_value(j));
    }

    // Ghost detection without seen(N): sort+unique on colind
    std::vector<int> cols = colind;
    std::sort(cols.begin(), cols.end());
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());

    std::vector<int> ghost_cols;
    ghost_cols.reserve(cols.size());
    for (int j : cols) {
        if (j < 0 || j >= N) continue;
        if ((j % P) != rank) ghost_cols.push_back(j);
    }

    // Group requests by owner
    std::vector<int> sendcounts(P, 0);
    for (int j : ghost_cols) sendcounts[j % P]++;

    std::vector<int> sdispls(P, 0);
    for (int p = 1; p < P; ++p)
        sdispls[p] = sdispls[p - 1] + sendcounts[p - 1];

    std::vector<int> sendbuf((size_t)ghost_cols.size());
    {
        std::vector<int> tmp = sdispls;
        for (int j : ghost_cols) {
            int p = j % P;
            sendbuf[(size_t)tmp[p]++] = j;
        }
    }

    // Exchange counts
    std::vector<int> recvcounts(P, 0);
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT,
                 recvcounts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    std::vector<int> rdispls(P, 0);
    int tot_recv = 0;
    for (int p = 0; p < P; ++p) {
        rdispls[p] = tot_recv;
        tot_recv += recvcounts[p];
    }

    // Receive requests
    std::vector<int> recvbuf((size_t)tot_recv);
    MPI_Alltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), MPI_INT,
                  recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // Prepare responses (values for requested columns)
    std::vector<double> sendvals((size_t)tot_recv);
    for (int i = 0; i < tot_recv; ++i) {
        int j = recvbuf[(size_t)i];
        sendvals[(size_t)i] = x_value(j);
    }

    // Receive values for my requests
    std::vector<double> recvvals(sendbuf.size());
    MPI_Alltoallv(sendvals.data(), recvcounts.data(), rdispls.data(), MPI_DOUBLE,
                  recvvals.data(), sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                  MPI_COMM_WORLD);

    // Append ghosts in same order as sendbuf
    size_t idx = 0;
    for (int p = 0; p < P; ++p) {
        for (int k = 0; k < sendcounts[p]; ++k) {
            int j = sendbuf[(size_t)sdispls[p] + (size_t)k];
            if (g2l[j] == -1) {
                g2l[j] = (int)x_local.size();
                x_local.push_back(recvvals[idx]);
            }
            idx++;
        }
    }
}

// ============================================================
// OPTION B: OpenMP schedule(runtime) + omp_set_schedule()
// ============================================================

static omp_sched_t parse_omp_schedule(const std::string& schedule, int& chunk_io)
{
    std::string s = lower_copy(schedule);
    if (chunk_io < 1) chunk_io = 1;

    if (s == "dynamic") return omp_sched_dynamic;
    if (s == "guided")  return omp_sched_guided;
    if (s == "auto")    return omp_sched_auto;
    if (s == "static")  return omp_sched_static;

    return omp_sched_static;
}

static const char* omp_sched_name(omp_sched_t k)
{
    switch (k) {
        case omp_sched_static:  return "static";
        case omp_sched_dynamic: return "dynamic";
        case omp_sched_guided:  return "guided";
        case omp_sched_auto:    return "auto";
        default:                return "unknown";
    }
}

// ============================================================
// SpMV local (OpenMP) using schedule(runtime) (chunk is real!)
// ============================================================

static void spmv_csr_local_omp_runtime(const CSRLocal& A,
                                      const std::vector<double>& x_local,
                                      const std::vector<int>& g2l,
                                      std::vector<double>& y_local)
{
    y_local.assign((size_t)A.localM, 0.0);

    #pragma omp parallel for schedule(runtime)
    for (int r = 0; r < A.localM; r++) {
        double sum = 0.0;
        for (int k = A.rowptr[r]; k < A.rowptr[r + 1]; k++) {
            int gc  = A.col[k];
            int idx = g2l[gc];

#if SPMV_CHECK_G2L
            if (idx < 0 || idx >= (int)x_local.size()) {
                #pragma omp critical
                { std::cerr << "[fatal] missing/invalid g2l for col " << gc << "\n"; }
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
#endif
            sum += A.val[k] * x_local[idx];
        }
        y_local[(size_t)r] = sum;
    }
}

// ============================================================
// Percentile90 on rank0 (like D1) from samples (ms)
// ============================================================

static double percentile90_ms_from_samples(std::vector<double>& samples)
{
    if (samples.empty()) return 0.0;
    size_t k = (size_t)std::ceil(0.90 * (double)samples.size());
    if (k == 0) k = 1;
    --k;
    std::nth_element(samples.begin(), samples.begin() + k, samples.end());
    return samples[k];
}

// ============================================================
// Validation: gather y + gather COO then compute rel L2 and max abs on rank0
// ============================================================

static ValidationResult validate_spmv_mpi(int rank, int P,
                                         int M, int N,
                                         const std::vector<COOEntry>& coo_local,
                                         const std::vector<double>& y_local)
{
    // Gather y to rank0 as (global_row, y_val)
    const int localM = (int)y_local.size();
    std::vector<int> global_rows(localM);
    for (int lr = 0; lr < localM; lr++)
        global_rows[lr] = rank + lr * P;

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

    MPI_Gatherv((void*)y_local.data(), localM, MPI_DOUBLE,
                rank == 0 ? all_yvals.data() : nullptr,
                rank == 0 ? recv_counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Gather COO to rank0
    int local_nnz = (int)coo_local.size();
    std::vector<int> nnz_counts, nnz_displs;
    if (rank == 0) {
        nnz_counts.resize(P, 0);
        nnz_displs.resize(P, 0);
    }

    MPI_Gather(&local_nnz, 1, MPI_INT,
               rank == 0 ? nnz_counts.data() : nullptr, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    int tot_nnz = 0;
    std::vector<COOEntry> coo_all;
    if (rank == 0) {
        for (int p=0;p<P;p++){
            nnz_displs[p] = tot_nnz;
            tot_nnz += nnz_counts[p];
        }
        coo_all.resize((size_t)tot_nnz);
    }

    MPI_Datatype MPI_COO = make_mpi_coo_type();

    MPI_Gatherv((void*)coo_local.data(), local_nnz, MPI_COO,
                rank == 0 ? (void*)coo_all.data() : nullptr,
                rank == 0 ? nnz_counts.data() : nullptr,
                rank == 0 ? nnz_displs.data() : nullptr,
                MPI_COO, 0, MPI_COMM_WORLD);

    MPI_Type_free(&MPI_COO);

    ValidationResult res{};

    if (rank == 0) {
        // Build y (global)
        std::vector<double> y(M, 0.0);
        for (int idx = 0; idx < total_rows; idx++) {
            int gi = all_grows[idx];
            if (0 <= gi && gi < M) y[gi] = all_yvals[idx];
        }

        // Build serial CSR from COO (simple counting)
        std::vector<int> rowptr(M+1, 0);
        for (const auto& e : coo_all) rowptr[(int)e.i + 1]++;
        for (int i=0;i<M;i++) rowptr[i+1] += rowptr[i];

        std::vector<int> col((size_t)coo_all.size(), 0);
        std::vector<double> val((size_t)coo_all.size(), 0.0);
        std::vector<int> cur = rowptr;
        for (const auto& e : coo_all) {
            int r = (int)e.i;
            int p = cur[r]++;
            col[(size_t)p] = (int)e.j;
            val[(size_t)p] = e.v;
        }

        // Compute yref = A*x with same x_value(j)
        std::vector<double> yref(M, 0.0);
        for (int i=0;i<M;i++){
            double acc = 0.0;
            for (int k=rowptr[i]; k<rowptr[i+1]; k++){
                acc += val[(size_t)k] * x_value(col[(size_t)k]);
            }
            yref[i] = acc;
        }

        long double diff2=0.0L, ref2=0.0L, maxa=0.0L;
        for (int i=0;i<M;i++){
            long double d = (long double)y[i] - (long double)yref[i];
            diff2 += d*d;
            ref2  += (long double)yref[i]*(long double)yref[i];
            long double ad = fabsl(d);
            if (ad > maxa) maxa = ad;
        }
        long double rel = (ref2>0.0L) ? sqrtl(diff2/ref2) : sqrtl(diff2);

        res.rel_L2_error = rel;
        res.max_abs_error = maxa;
    }

    return res;
}

// ============================================================
// Logging (human readable + TSV) like D1
// ============================================================

static void append_log_rank0(const char* argv0,
                            const std::string& mtx,
                            int M, int N, long long nz,
                            int ranks,
                            int threads,
                            const std::string& schedule,
                            int chunk,
                            int warmup,
                            int repeats,
                            int trials,
                            int sort_rows,
                            int do_validation,
                            const ValidationResult& vres,
                            double p90_ms,
                            double gflops,
                            double gbps)
{
    try {
        fs::path root = repo_root_from_exe(argv0);
        fs::path results_dir = root / "results";
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

        auto now   = std::chrono::system_clock::now();
        std::time_t t_c = std::chrono::system_clock::to_time_t(now);
        std::tm* tm = std::localtime(&t_c);

        std::ostringstream validation_line;
        validation_line.setf(std::ios::scientific, std::ios::floatfield);
        validation_line << "rel_L2_error=" << (long double)vres.rel_L2_error
                        << ", max_abs_error=" << (long double)vres.max_abs_error;

        fout << "=================================================================\n";
        fout << "Run at " << std::put_time(tm, "%Y-%m-%d %H:%M:%S") << "\n";
        fout << "Matrix    : " << mtx
             << " (" << M << " x " << N
             << ", nnz(header) = " << nz << ")\n";
        fout << "Config    : ranks=" << ranks
             << ", threads=" << threads
             << ", schedule=" << lower_copy(schedule)
             << "(chunk=" << chunk << ")"
             << ", warmup=" << warmup
             << ", repeats=" << repeats
             << ", trials=" << trials
             << ", sort_rows=" << sort_rows
             << ", validation=" << do_validation << "\n";
        fout << "I/O       : MPI-IO chunk parsing (coordinate real), overlap boundary handling\n";
        if (do_validation) fout << "Validation: " << validation_line.str() << "\n";
        else              fout << "Validation: (skipped)\n";

        fout << std::fixed << std::setprecision(3);
        fout << "Results   : p90_ms=" << p90_ms
             << ", GFLOPS=" << gflops
             << ", GBps=" << gbps << "\n";
        fout << "=================================================================\n\n";
    } catch (const std::exception& e) {
        std::cerr << "[warning] Exception while writing results file: " << e.what() << "\n";
    }

    // TSV one-line per run
    try {
        fs::path root = repo_root_from_exe(argv0);
        fs::path results_dir = root / "results";
        fs::create_directories(results_dir);

        fs::path tsv_path = results_dir / "spmv_mpi_results.tsv";
        bool write_header = !fs::exists(tsv_path);

        std::ofstream fout(tsv_path, std::ios::app);
        if (!fout) {
            std::cerr << "[warning] Could not write TSV to: " << tsv_path << "\n";
            return;
        }

        if (write_header) {
            fout << "timestamp\tmatrix\tM\tN\tnz\tranks\tthreads\tsched\tchunk\twarmup\trepeats\ttrials\tsort_rows\tp90_ms\tgflops\tgbps\tvalidation\trelL2\tmaxAbs\n";
        }

        auto now   = std::chrono::system_clock::now();
        std::time_t t_c = std::chrono::system_clock::to_time_t(now);
        std::tm* tm = std::localtime(&t_c);

        std::ostringstream ts;
        ts << std::put_time(tm, "%Y-%m-%d_%H:%M:%S");

        fout << ts.str() << "\t"
             << mtx << "\t"
             << M << "\t" << N << "\t" << nz << "\t"
             << ranks << "\t" << threads << "\t"
             << lower_copy(schedule) << "\t" << chunk << "\t"
             << warmup << "\t" << repeats << "\t" << trials << "\t"
             << sort_rows << "\t"
             << std::setprecision(9) << p90_ms << "\t"
             << std::setprecision(9) << gflops << "\t"
             << std::setprecision(9) << gbps << "\t"
             << do_validation << "\t"
             << (do_validation ? (double)vres.rel_L2_error : 0.0) << "\t"
             << (do_validation ? (double)vres.max_abs_error : 0.0) << "\n";
    } catch (...) {
        // ignore
    }
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank=0, P=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " <matrix.mtx> <threads> [static|dynamic|guided] [chunk] [repeats] [trials] [--no-validate] [--sort-rows]\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Parse args (D1-like)
    std::string mtxArg = argv[1];
    const int threads_req = std::max(1, std::atoi(argv[2]));
    const std::string schedule_arg = (argc >= 4 ? std::string(argv[3]) : std::string("static"));
    const int chunk_arg   = (argc >= 5 ? std::max(1, std::atoi(argv[4])) : 64);
    const int repeats = (argc >= 6 ? std::max(1, std::atoi(argv[5])) : 10);
    const int trials  = (argc >= 7 ? std::max(1, std::atoi(argv[6])) : 5);
    const int warmup  = 2;

    int do_validation = 1;
    int sort_rows = 0;
    if (rank == 0) {
        for (int a=1;a<argc;a++){
            std::string s = argv[a];
            if (s == "--no-validate") do_validation = 0;
            if (s == "--sort-rows")   sort_rows = 1;
        }
    }
    MPI_Bcast(&do_validation, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sort_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resolve matrix path on rank0 like D1, then broadcast
    std::string mtxResolved;
    if (rank == 0) {
        mtxResolved = resolve_matrix_path_rank0(mtxArg);
        if (mtxResolved.empty()) {
            std::cerr << "[fatal] cannot open '" << mtxArg << "' in any of the expected locations:\n"
                      << "  - " << mtxArg << "\n"
                      << "  - matrices/" << mtxArg << "\n"
                      << "  - bin/matrices/" << mtxArg << "\n"
                      << "  - ../bin/matrices/" << mtxArg << "\n"
                      << "  - $MATRICES_DIR/" << mtxArg << " (if set)\n";
        }
    }
    int ok = (rank==0 ? (!mtxResolved.empty()) : 0);
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok) {
        MPI_Finalize();
        return 1;
    }
    mtxResolved = bcast_string_from_rank0(mtxResolved, rank);

    // Set OpenMP config
    omp_set_dynamic(0);
    omp_set_num_threads(threads_req);

    // OPTION B: make chunk REAL via schedule(runtime)
    int chunk_eff = chunk_arg;
    omp_sched_t sched_eff = parse_omp_schedule(schedule_arg, chunk_eff);
    omp_set_schedule(sched_eff, chunk_eff);

    // Read matrix (BONUS 4 MPI-IO) -> COO local
    int M=0, N=0, nz_header=0;
    std::vector<COOEntry> coo_local;
    parallel_read_matrix_market_mpiio(mtxResolved.c_str(), rank, P, M, N, nz_header, coo_local);

    // Build CSR local
    CSRLocal A = coo_to_csr_cyclic_rows(coo_local, M, rank, P);
    if (sort_rows) sort_csr_rows_by_col(A);

    // Build x_local + ghosts (BONUS 1+2)
    std::vector<double> x_local;
    std::vector<int> g2l;
    build_x_and_exchange_ghosts_alltoallv(rank, P, N, A.col, x_local, g2l);

    // Warmup
    std::vector<double> y_local;
    for (int w=0; w<warmup; ++w) {
        spmv_csr_local_omp_runtime(A, x_local, g2l, y_local);
    }

    // Timing samples (P90 like D1), measure MAX over ranks each sample
    std::vector<double> samples_ms;
    if (rank == 0) samples_ms.reserve((size_t)repeats * (size_t)trials);

    for (int t=0; t<trials; ++t) {
        for (int r=0; r<repeats; ++r) {
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            spmv_csr_local_omp_runtime(A, x_local, g2l, y_local);
            double t1 = MPI_Wtime();

            double local_ms = (t1 - t0) * 1000.0;
            double max_ms = 0.0;
            MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            if (rank == 0) samples_ms.push_back(max_ms);
        }
    }

    double p90_ms = 0.0;
    if (rank == 0) {
        p90_ms = percentile90_ms_from_samples(samples_ms);
    }

    // Validation (optional, collectives safe: all or none)
    ValidationResult vres{};
    if (do_validation) {
        vres = validate_spmv_mpi(rank, P, M, N, coo_local, y_local);
    }

    // Compute perf numbers on rank0 using nz from header (consistent)
    if (rank == 0) {
        const long long nnz = (long long)nz_header;

        double gflops = (p90_ms > 0.0)
            ? (2.0 * (double)nnz) / (p90_ms / 1000.0) / 1e9
            : std::numeric_limits<double>::infinity();

        // Simple byte model (similar spirit to D1)
        double bytes = (double)nnz * (8.0 + 4.0 + 8.0) + (double)M * 8.0;
        double gbps = (p90_ms > 0.0)
            ? bytes / (p90_ms / 1000.0) / 1e9
            : std::numeric_limits<double>::infinity();

        // Real used threads
        int used_threads = omp_used_threads();

        // Read back the effective OpenMP schedule (truth source)
        omp_sched_t sched_now;
        int chunk_now;
        omp_get_schedule(&sched_now, &chunk_now);

        // Output (D1 style)
        std::cout << std::fixed << std::setprecision(3);

        std::cout << "\n=================================================================\n";
        std::cout << "                 SpMV Benchmark (MPI + OpenMP)\n";
        std::cout << "=================================================================\n\n";

        std::cout << "MATRIX INFO\n";
        std::cout << "  File                 : " << mtxResolved << "\n";
        std::cout << "  Dimensions           : " << M << " x " << N << "\n";
        std::cout << "  Non-zero entries     : " << nnz << "\n\n";

        std::cout << "BENCHMARK SETTINGS\n";
        std::cout << "  MPI ranks            : " << P << "\n";
        std::cout << "  Threads              : " << used_threads << "\n";
        std::cout << "  Schedule             : " << omp_sched_name(sched_now)
                  << " (chunk = " << chunk_now << ")\n";
        std::cout << "  Warmup runs          : " << warmup << "\n";
        std::cout << "  Repeats per trial    : " << repeats << "\n";
        std::cout << "  Number of trials     : " << trials << "\n";
        std::cout << "  Sort CSR rows        : " << (sort_rows ? "yes" : "no") << "\n";
        std::cout << "  I/O                  : MPI-IO chunk parsing (Bonus 4)\n";
        std::cout << "  Time metric          : 90th percentile (P90) of max-rank time\n\n";

        // Validation on stderr (scientific) like D1
        if (do_validation) {
            auto old_prec  = std::cerr.precision();
            auto old_flags = std::cerr.flags();

            std::cerr.setf(std::ios::scientific, std::ios::floatfield);
            std::cerr << "[validation] relative_L2_error = " << vres.rel_L2_error
                      << "   max_absolute_error = " << vres.max_abs_error << "\n";

            std::cerr.flags(old_flags);
            std::cerr.precision(old_prec);
        } else {
            std::cerr << "[validation] skipped\n";
        }

        std::cout << "\nRESULTS\n";
        std::cout << "  P90 execution time   : " << p90_ms << " ms\n";
        std::cout << "  Throughput           : " << gflops << " GFLOPS\n";
        std::cout << "  Estimated bandwidth  : " << gbps << " GB/s\n\n";
        std::cout << "=================================================================\n\n";

        // Write results like D1 (block log + TSV)
        append_log_rank0(argv[0], mtxResolved, M, N, nnz, P, used_threads,
                         omp_sched_name(sched_now), chunk_now, warmup, repeats, trials,
                         sort_rows, do_validation, vres,
                         p90_ms, gflops, gbps);

        fs::path root = repo_root_from_exe(argv[0]);
        std::cout << "[Rank0] Results appended to " << (root / "results" / "spmv_mpi_results.txt") << "\n";
    }

    MPI_Finalize();
    return 0;
}
