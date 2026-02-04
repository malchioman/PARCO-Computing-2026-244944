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
#include <numeric>

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

// Ghost exchange plan (precomputed once, used every iteration)
struct GhostPlan {
    int P = 1;

    // My requests to owners (columns I need)
    std::vector<int> sendcounts;   // size P
    std::vector<int> sdispls;      // size P
    std::vector<int> sendcols;     // size sum(sendcounts)
    std::vector<int> ghost_lidx;   // same size as sendcols: where to write received values in x_local

    // Requests received from others (columns others need from me)
    std::vector<int> recvcounts;   // size P
    std::vector<int> rdispls;      // size P
    std::vector<int> recvcols;     // size sum(recvcounts)

    // Preallocated per-iteration value buffers (no alloc in timing loop)
    std::vector<double> tmp_sendvals; // size recvcols.size()
    std::vector<double> tmp_recvvals; // size sendcols.size()

    bool has_ghosts() const { return !sendcols.empty() || !recvcols.empty(); }
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
    // exe in <repo>/bin/spmv_mpi -> root = parent(bin)
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
    const double inv = 1.0 / 9007199254740992.0; // 2^53
    double u = ((x >> 11) & 0x1fffffffffffffULL) * inv;
    if (u <= 0.0) u = inv;
    if (u >= 1.0) u = 1.0 - inv;
    return u;
}

static inline double x_value(int j) {
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
    return (gi - rank) / P;
}

static inline int num_local_rows_cyclic(int M, int rank, int P) {
    if (rank >= M) return 0;
    return 1 + (M - 1 - rank) / P;
}

// ============================================================
// MPI datatype for COOEntry
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
// MatrixMarket parsing
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

// ============================================================
// BONUS 4: parallel matrix reading with MPI-IO chunk parsing
// ============================================================

static void parallel_read_matrix_market_mpiio(const char* path,
                                              int rank, int P,
                                              int& M, int& N, int& nz_header,
                                              std::vector<COOEntry>& coo_chunk)
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

    long long tmp = 0;
    if (rank == 0) tmp = (long long)data_start;
    MPI_Bcast(&tmp, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    data_start = (MPI_Offset)tmp;

    M = M0; N = N0; nz_header = nz0;

    const MPI_Offset data_bytes = fsize - data_start;
    const MPI_Offset chunk = (data_bytes + P - 1) / P;

    MPI_Offset my_start = data_start + (MPI_Offset)rank * chunk;
    MPI_Offset my_end   = std::min<MPI_Offset>(data_start + (MPI_Offset)(rank + 1) * chunk, fsize);

    const MPI_Offset OVER = 4096;
    MPI_Offset read_start = (rank == 0) ? my_start : std::max<MPI_Offset>(data_start, my_start - OVER);
    MPI_Offset read_end   = (rank == P - 1) ? my_end   : std::min<MPI_Offset>(fsize, my_end + OVER);

    MPI_Offset read_len = read_end - read_start;

    std::vector<char> raw((size_t)read_len, '\0');
    mpi_file_read_at_all_big(fh, read_start, raw.data(), read_len);

    MPI_File_close(&fh);

    std::string s(raw.data(), raw.size());

    MPI_Offset rel_my_start = my_start - read_start;
    MPI_Offset rel_my_end   = my_end   - read_start;

    size_t a = (size_t)std::max<MPI_Offset>(0, rel_my_start);
    size_t b = (size_t)std::max<MPI_Offset>(0, rel_my_end);
    if (a > s.size()) a = s.size();
    if (b > s.size()) b = s.size();

    if (rank != 0) {
        while (a < s.size() && s[a] != '\n') a++;
        if (a < s.size()) a++;
    }
    if (rank != P - 1) {
        while (b > 0 && s[b - 1] != '\n') b--;
    }

    coo_chunk.clear();
    if (b <= a) return;

    std::istringstream iss(s.substr(a, b - a));
    std::string line;

    coo_chunk.reserve((size_t)std::max(1, nz_header / P));

    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        line = rtrim_cr(line);
        if (line.empty()) continue;
        if (!line.empty() && line[0] == '%') continue;

        int i0, j0;
        double v;
        if (!parse_triplet_line(line, i0, j0, v)) continue;
        if (i0 < 0 || i0 >= M || j0 < 0 || j0 >= N) continue;

        coo_chunk.push_back({(int32_t)i0, (int32_t)j0, v});
    }
}

// ============================================================
// Redistribute COO triplets by owner row (cyclic i%P)
// ============================================================

static void redistribute_coo_by_row_owner(std::vector<COOEntry>& coo_chunk,
                                          int /*rank*/, int P)
{
    std::vector<int> sendcounts(P, 0);
    for (const auto& e : coo_chunk) {
        int dst = owner_row((int)e.i, P);
        sendcounts[dst]++;
    }

    std::vector<int> sdispls(P, 0);
    for (int p = 1; p < P; ++p) sdispls[p] = sdispls[p-1] + sendcounts[p-1];

    std::vector<COOEntry> sendbuf(coo_chunk.size());
    {
        std::vector<int> cursor = sdispls;
        for (const auto& e : coo_chunk) {
            int dst = owner_row((int)e.i, P);
            sendbuf[(size_t)cursor[dst]++] = e;
        }
    }

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

    std::vector<COOEntry> recvbuf((size_t)tot_recv);

    MPI_Datatype MPI_COO = make_mpi_coo_type();
    MPI_Alltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), MPI_COO,
                  recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_COO,
                  MPI_COMM_WORLD);
    MPI_Type_free(&MPI_COO);

    coo_chunk.swap(recvbuf);
}

// ============================================================
// COO -> CSR local (cyclic rows)
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

// Optional sort by column within each CSR row
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
// OpenMP schedule(runtime) + omp_set_schedule()
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
// SpMV local (OpenMP) using schedule(runtime)
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
            int idx = g2l[(size_t)gc];

#if SPMV_CHECK_G2L
            if (idx < 0 || idx >= (int)x_local.size()) {
                #pragma omp critical
                { std::cerr << "[fatal] missing/invalid g2l for col " << gc << "\n"; }
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
#endif
            sum += A.val[k] * x_local[(size_t)idx];
        }
        y_local[(size_t)r] = sum;
    }
}

// ============================================================
// Percentile90 on rank0 (like D1)
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
// Ghost plan build + per-iteration exchange (values only)
// ============================================================

static void build_ghost_plan_and_x(int rank, int P, int N,
                                   const std::vector<int>& colind,
                                   std::vector<double>& x_local,
                                   std::vector<int>& g2l,
                                   GhostPlan& plan)
{
    plan = GhostPlan{};
    plan.P = P;

    g2l.assign((size_t)N, -1);
    x_local.clear();
    x_local.reserve((size_t)(N / std::max(1, P)) + 1024);

    // Owned x entries (cyclic by column owner = j%P)
    for (int j = rank; j < N; j += P) {
        g2l[(size_t)j] = (int)x_local.size();
        x_local.push_back(x_value(j));
    }

    // Unique columns touched by local CSR
    std::vector<int> cols = colind;
    std::sort(cols.begin(), cols.end());
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());

    std::vector<int> ghost_cols;
    ghost_cols.reserve(cols.size());
    for (int j : cols) {
        if (j < 0 || j >= N) continue;
        if ((j % P) != rank) ghost_cols.push_back(j);
    }

    // Requests grouped by owner
    plan.sendcounts.assign(P, 0);
    for (int j : ghost_cols) plan.sendcounts[j % P]++;

    plan.sdispls.assign(P, 0);
    for (int p = 1; p < P; ++p)
        plan.sdispls[p] = plan.sdispls[p - 1] + plan.sendcounts[p - 1];

    plan.sendcols.resize((size_t)ghost_cols.size());
    {
        std::vector<int> cursor = plan.sdispls;
        for (int j : ghost_cols) {
            int owner = j % P;
            plan.sendcols[(size_t)cursor[owner]++] = j;
        }
    }

    // Allocate ghost slots in x_local + fill ghost_lidx (same order as sendcols)
    plan.ghost_lidx.resize(plan.sendcols.size());
    for (size_t k = 0; k < plan.sendcols.size(); ++k) {
        int j = plan.sendcols[k];
        if (g2l[(size_t)j] == -1) {
            g2l[(size_t)j] = (int)x_local.size();
            x_local.push_back(0.0); // placeholder, will be filled by exchange
        }
        plan.ghost_lidx[k] = g2l[(size_t)j];
    }

    // Exchange counts: recvcounts[p] = how many columns rank p asks from me
    plan.recvcounts.assign(P, 0);
    MPI_Alltoall(plan.sendcounts.data(), 1, MPI_INT,
                 plan.recvcounts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    plan.rdispls.assign(P, 0);
    int tot_recv = 0;
    for (int p = 0; p < P; ++p) {
        plan.rdispls[p] = tot_recv;
        tot_recv += plan.recvcounts[p];
    }
    plan.recvcols.resize((size_t)tot_recv);

    // Receive requested column IDs (others -> me)
    MPI_Alltoallv(plan.sendcols.data(), plan.sendcounts.data(), plan.sdispls.data(), MPI_INT,
                  plan.recvcols.data(), plan.recvcounts.data(), plan.rdispls.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // Preallocate per-iteration value buffers (IMPORTANT for clean timing)
    plan.tmp_sendvals.assign(plan.recvcols.size(), 0.0);
    plan.tmp_recvvals.assign(plan.sendcols.size(), 0.0);
}

static inline void exchange_ghost_values(GhostPlan& plan,
                                        const std::vector<int>& g2l,
                                        std::vector<double>& x_local)
{
    if (!plan.has_ghosts()) return;

    // Fill sendvals: values for columns others requested from me
    for (size_t i = 0; i < plan.recvcols.size(); ++i) {
        int j = plan.recvcols[i];
        int idx = g2l[(size_t)j];
#if SPMV_CHECK_G2L
        if (idx < 0 || idx >= (int)x_local.size()) {
            std::cerr << "[fatal] exchange_ghost_values: missing owned g2l for col " << j << "\n";
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
#endif
        plan.tmp_sendvals[i] = x_local[(size_t)idx];
    }

    // Receive values for my requested ghost columns (same order as sendcols)
    MPI_Alltoallv(plan.tmp_sendvals.data(), plan.recvcounts.data(), plan.rdispls.data(), MPI_DOUBLE,
                  plan.tmp_recvvals.data(), plan.sendcounts.data(), plan.sdispls.data(), MPI_DOUBLE,
                  MPI_COMM_WORLD);

    // Write ghosts into x_local at precomputed local indices
    for (size_t k = 0; k < plan.tmp_recvvals.size(); ++k) {
        int lidx = plan.ghost_lidx[k];
        x_local[(size_t)lidx] = plan.tmp_recvvals[k];
    }
}

// ============================================================
// Memory footprint + comm volume helpers
// ============================================================

template <typename T>
static inline long long bytes_of_vec(const std::vector<T>& v) {
    // model: capacity * sizeof(T) (closer to real allocation than size)
    return (long long)v.capacity() * (long long)sizeof(T);
}

static inline long long memory_bytes_csr(const CSRLocal& A) {
    return bytes_of_vec(A.rowptr) + bytes_of_vec(A.col) + bytes_of_vec(A.val);
}

static inline long long memory_bytes_plan(const GhostPlan& plan) {
    return bytes_of_vec(plan.sendcounts) + bytes_of_vec(plan.sdispls) + bytes_of_vec(plan.sendcols) + bytes_of_vec(plan.ghost_lidx)
         + bytes_of_vec(plan.recvcounts) + bytes_of_vec(plan.rdispls) + bytes_of_vec(plan.recvcols)
         + bytes_of_vec(plan.tmp_sendvals) + bytes_of_vec(plan.tmp_recvvals);
}

// per-iteration comm volume (values exchange only)
static inline void comm_bytes_per_iter_values_only(const GhostPlan& plan,
                                                   long long& sent_bytes,
                                                   long long& recv_bytes)
{
    // Each iter:
    //  - I send responses for recvcols -> sendvals size = recvcols.size() doubles
    //  - I receive values for my ghosts -> recvvals size = sendcols.size() doubles
    sent_bytes = (long long)plan.recvcols.size() * 8LL;
    recv_bytes = (long long)plan.sendcols.size() * 8LL;
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

    // Make chunk REAL via schedule(runtime)
    int chunk_eff = chunk_arg;
    omp_sched_t sched_eff = parse_omp_schedule(schedule_arg, chunk_eff);
    omp_set_schedule(sched_eff, chunk_eff);

    // Read matrix (MPI-IO) -> COO CHUNK (not yet owned by row)
    int M=0, N=0, nz_header=0;
    std::vector<COOEntry> coo_local;
    parallel_read_matrix_market_mpiio(mtxResolved.c_str(), rank, P, M, N, nz_header, coo_local);

    // Redistribute by row owner so coo_local becomes correct local COO
    redistribute_coo_by_row_owner(coo_local, rank, P);

    // Build CSR local
    CSRLocal A = coo_to_csr_cyclic_rows(coo_local, M, rank, P);
    if (sort_rows) sort_csr_rows_by_col(A);

    // Compute nnz_used (sum of actual nnz across ranks) for correct perf numbers
    long long local_nnz = (long long)A.val.size();
    long long nnz_used  = 0;
    MPI_Allreduce(&local_nnz, &nnz_used, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    // Build x_local + ghost plan (precompute), then do one initial exchange
    std::vector<double> x_local;
    std::vector<int> g2l;
    GhostPlan plan;
    build_ghost_plan_and_x(rank, P, N, A.col, x_local, g2l, plan);
    exchange_ghost_values(plan, g2l, x_local); // fill ghosts once before warmup/timing

    // Warmup: do BOTH (comm + compute)
    std::vector<double> y_local;
    for (int w=0; w<warmup; ++w) {
        exchange_ghost_values(plan, g2l, x_local);
        spmv_csr_local_omp_runtime(A, x_local, g2l, y_local);
    }

    // ============================================================
    // Memory footprint + comm volume measurements (model-based)
    // ============================================================

    // local memory (bytes)
    long long mem_csr   = memory_bytes_csr(A);
    long long mem_x     = bytes_of_vec(x_local);
    long long mem_y     = bytes_of_vec(y_local);
    long long mem_g2l   = bytes_of_vec(g2l);
    long long mem_plan  = memory_bytes_plan(plan);

    long long mem_total = mem_csr + mem_x + mem_y + mem_g2l + mem_plan;

    // comm volume per iteration (values exchange only)
    long long comm_sent_B = 0, comm_recv_B = 0;
    comm_bytes_per_iter_values_only(plan, comm_sent_B, comm_recv_B);
    long long comm_tot_B = comm_sent_B + comm_recv_B;

    // Reduce memory stats to rank0
    double mem_total_d = (double)mem_total;
    double mem_csr_d   = (double)mem_csr;
    double mem_x_d     = (double)mem_x;
    double mem_y_d     = (double)mem_y;
    double mem_g2l_d   = (double)mem_g2l;
    double mem_plan_d  = (double)mem_plan;

    double mem_total_sum=0, mem_total_max=0;
    double mem_csr_sum=0,   mem_csr_max=0;
    double mem_x_sum=0,     mem_x_max=0;
    double mem_y_sum=0,     mem_y_max=0;
    double mem_g2l_sum=0,   mem_g2l_max=0;
    double mem_plan_sum=0,  mem_plan_max=0;

    MPI_Reduce(&mem_total_d, &mem_total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mem_total_d, &mem_total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&mem_csr_d,   &mem_csr_sum,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mem_csr_d,   &mem_csr_max,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&mem_x_d,     &mem_x_sum,     1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mem_x_d,     &mem_x_max,     1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&mem_y_d,     &mem_y_sum,     1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mem_y_d,     &mem_y_max,     1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&mem_g2l_d,   &mem_g2l_sum,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mem_g2l_d,   &mem_g2l_max,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&mem_plan_d,  &mem_plan_sum,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mem_plan_d,  &mem_plan_max,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Reduce comm stats to rank0
    double comm_sent_d = (double)comm_sent_B;
    double comm_recv_d = (double)comm_recv_B;
    double comm_tot_d  = (double)comm_tot_B;

    double comm_sent_sum=0, comm_sent_max=0;
    double comm_recv_sum=0, comm_recv_max=0;
    double comm_tot_sum=0,  comm_tot_max=0;

    MPI_Reduce(&comm_sent_d, &comm_sent_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_sent_d, &comm_sent_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&comm_recv_d, &comm_recv_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_recv_d, &comm_recv_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&comm_tot_d,  &comm_tot_sum,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_tot_d,  &comm_tot_max,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ============================================================
    // Timing samples: comm-only, compute-only, end-to-end
    // ============================================================

    std::vector<double> samples_e2e_ms;
    std::vector<double> samples_comp_ms;
    std::vector<double> samples_comm_ms;

    if (rank == 0) {
        size_t cap = (size_t)repeats * (size_t)trials;
        samples_e2e_ms.reserve(cap);
        samples_comp_ms.reserve(cap);
        samples_comm_ms.reserve(cap);
    }

    for (int t=0; t<trials; ++t) {
        for (int r=0; r<repeats; ++r) {
            MPI_Barrier(MPI_COMM_WORLD);

            double t0 = MPI_Wtime();
            exchange_ghost_values(plan, g2l, x_local);
            double t1 = MPI_Wtime();
            spmv_csr_local_omp_runtime(A, x_local, g2l, y_local);
            double t2 = MPI_Wtime();

            double local_comm_ms = (t1 - t0) * 1000.0;
            double local_comp_ms = (t2 - t1) * 1000.0;
            double local_e2e_ms  = (t2 - t0) * 1000.0;

            double max_comm_ms = 0.0, max_comp_ms = 0.0, max_e2e_ms = 0.0;
            MPI_Reduce(&local_comm_ms, &max_comm_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_comp_ms, &max_comp_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_e2e_ms,  &max_e2e_ms,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                samples_comm_ms.push_back(max_comm_ms);
                samples_comp_ms.push_back(max_comp_ms);
                samples_e2e_ms.push_back(max_e2e_ms);
            }
        }
    }

    double p90_e2e_ms  = 0.0;
    double p90_comp_ms = 0.0;
    double p90_comm_ms = 0.0;

    if (rank == 0) {
        p90_e2e_ms  = percentile90_ms_from_samples(samples_e2e_ms);
        p90_comp_ms = percentile90_ms_from_samples(samples_comp_ms);
        p90_comm_ms = percentile90_ms_from_samples(samples_comm_ms);
    }

    // Validation placeholder
    ValidationResult vres{};

    // ============================================================
    // Output (keep parsing compatibility)
    // ============================================================

    if (rank == 0) {
        const long long nnz = nnz_used;

        auto gflops_from_ms = [&](double ms)->double {
            return (ms > 0.0)
                ? (2.0 * (double)nnz) / (ms / 1000.0) / 1e9
                : std::numeric_limits<double>::infinity();
        };

        auto gbps_from_ms = [&](double ms)->double {
            double bytes = (double)nnz * (8.0 + 4.0 + 8.0) + (double)M * 8.0;
            return (ms > 0.0)
                ? bytes / (ms / 1000.0) / 1e9
                : std::numeric_limits<double>::infinity();
        };

        // End-to-end (PRIMARY for scripts)
        double gflops_e2e = gflops_from_ms(p90_e2e_ms);
        double gbps_e2e   = gbps_from_ms(p90_e2e_ms);

        // Compute-only
        double gflops_comp = gflops_from_ms(p90_comp_ms);
        double gbps_comp   = gbps_from_ms(p90_comp_ms);

        int used_threads = omp_used_threads();

        omp_sched_t sched_now;
        int chunk_now;
        omp_get_schedule(&sched_now, &chunk_now);

        auto B_to_MiB = [&](double B)->double { return B / (1024.0 * 1024.0); };
        auto B_to_KiB = [&](double B)->double { return B / 1024.0; };

        double mem_total_avg = mem_total_sum / (double)P;
        double mem_csr_avg   = mem_csr_sum   / (double)P;
        double mem_x_avg     = mem_x_sum     / (double)P;
        double mem_y_avg     = mem_y_sum     / (double)P;
        double mem_g2l_avg   = mem_g2l_sum   / (double)P;
        double mem_plan_avg  = mem_plan_sum  / (double)P;

        double comm_sent_avg = comm_sent_sum / (double)P;
        double comm_recv_avg = comm_recv_sum / (double)P;
        double comm_tot_avg  = comm_tot_sum  / (double)P;

        std::cout << std::fixed << std::setprecision(3);

        std::cout << "\n=================================================================\n";
        std::cout << "                 SpMV Benchmark (MPI + OpenMP)\n";
        std::cout << "=================================================================\n\n";

        std::cout << "MATRIX INFO\n";
        std::cout << "  File                 : " << mtxResolved << "\n";
        std::cout << "  Dimensions           : " << M << " x " << N << "\n";
        std::cout << "  Non-zero entries     : " << nnz_used << " (header=" << (long long)nz_header << ")\n\n";

        std::cout << "BENCHMARK SETTINGS\n";
        std::cout << "  MPI ranks            : " << P << "\n";
        std::cout << "  Threads              : " << used_threads << "\n";
        std::cout << "  Schedule             : " << omp_sched_name(sched_now)
                  << " (chunk = " << chunk_now << ")\n";
        std::cout << "  Warmup runs          : " << warmup << "\n";
        std::cout << "  Repeats per trial    : " << repeats << "\n";
        std::cout << "  Number of trials     : " << trials << "\n";
        std::cout << "  Sort CSR rows        : " << (sort_rows ? "yes" : "no") << "\n";
        std::cout << "  I/O                  : MPI-IO chunk parsing (redistributed by row owner)\n";
        std::cout << "  Comm                 : ghost exchange (Alltoallv values) each iteration\n";
        std::cout << "  Time metric          : 90th percentile (P90) of max-rank time\n\n";

        std::cerr << "[validation] skipped\n";

        std::cout << "MEMORY FOOTPRINT (MODEL, based on vector capacity)\n";
        std::cout << "  Per-rank avg (MiB)   : total=" << B_to_MiB(mem_total_avg)
                  << "  [CSR=" << B_to_MiB(mem_csr_avg)
                  << ", x_local=" << B_to_MiB(mem_x_avg)
                  << ", y_local=" << B_to_MiB(mem_y_avg)
                  << ", g2l=" << B_to_MiB(mem_g2l_avg)
                  << ", plan/buffers=" << B_to_MiB(mem_plan_avg) << "]\n";
        std::cout << "  Per-rank max (MiB)   : total=" << B_to_MiB(mem_total_max)
                  << "  [CSR=" << B_to_MiB(mem_csr_max)
                  << ", x_local=" << B_to_MiB(mem_x_max)
                  << ", y_local=" << B_to_MiB(mem_y_max)
                  << ", g2l=" << B_to_MiB(mem_g2l_max)
                  << ", plan/buffers=" << B_to_MiB(mem_plan_max) << "]\n\n";

        std::cout << "COMMUNICATION VOLUME (VALUES ONLY, per iteration)\n";
        std::cout << "  Per-rank avg (KiB)   : sent=" << B_to_KiB(comm_sent_avg)
                  << ", recv=" << B_to_KiB(comm_recv_avg)
                  << ", total=" << B_to_KiB(comm_tot_avg) << "\n";
        std::cout << "  Per-rank max (KiB)   : sent=" << B_to_KiB(comm_sent_max)
                  << ", recv=" << B_to_KiB(comm_recv_max)
                  << ", total=" << B_to_KiB(comm_tot_max) << "\n";
        std::cout << "  Global total (KiB)   : sent=" << B_to_KiB(comm_sent_sum)
                  << ", recv=" << B_to_KiB(comm_recv_sum)
                  << ", total=" << B_to_KiB(comm_tot_sum) << "\n\n";

        std::cout << "RESULTS (END-TO-END: COMM + COMPUTE)\n";
        std::cout << "  P90 execution time   : " << p90_e2e_ms << " ms\n";
        std::cout << "  Throughput           : " << gflops_e2e << " GFLOPS\n";
        std::cout << "  Estimated bandwidth  : " << gbps_e2e << " GB/s\n\n";

        std::cout << "EXTRA RESULTS (COMPUTE-ONLY)\n";
        std::cout << "  Compute-only P90 time: " << p90_comp_ms << " ms\n";
        std::cout << "  Compute-only GFLOPS  : " << gflops_comp << " GFLOPS\n";
        std::cout << "  Compute-only BW      : " << gbps_comp << " GB/s\n\n";

        std::cout << "EXTRA RESULTS (COMM-ONLY)\n";
        std::cout << "  Comm-only P90 time   : " << p90_comm_ms << " ms\n\n";

        std::cout << "=================================================================\n\n";
    }

    MPI_Finalize();
    return 0;
}
