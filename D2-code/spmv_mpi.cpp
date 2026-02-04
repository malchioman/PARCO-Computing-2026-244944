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

#if !defined(_WIN32)
  #include <unistd.h>   // readlink
  #include <limits.h>   // PATH_MAX
#endif

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

    // Preallocated per-iteration buffers (no alloc in timing loop)
    std::vector<double> tmp_sendvals; // size recvcols.size()
    std::vector<double> tmp_recvvals; // size sendcols.size()

    bool has_ghosts() const { return !sendcols.empty() || !recvcols.empty(); }
};

// Pair used for gathering cyclic rows to rank0 during validation
struct RowVal {
    int32_t row;
    double  val;
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
    if (len > 0) MPI_Bcast(s.data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
    return s;
}

// Robust exe path (PBS-safe)
static fs::path exe_path_robust(const char* argv0) {
#if !defined(_WIN32)
    // On Linux, /proc/self/exe is the most reliable
    char buf[PATH_MAX];
    ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf)-1);
    if (n > 0) {
        buf[n] = '\0';
        return fs::path(buf);
    }
#endif
    // Fallbacks
    try {
        return fs::canonical(fs::path(argv0));
    } catch (...) {}
    try {
        return fs::absolute(fs::path(argv0));
    } catch (...) {}
    return fs::current_path() / fs::path(argv0);
}

static fs::path repo_root_from_exe(const char* argv0) {
    // exe in <repo>/bin/spmv_mpi -> root = parent(bin)
    try {
        fs::path exe = exe_path_robust(argv0);
        fs::path bin = exe.parent_path();
        return bin.parent_path();
    } catch (...) {
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
    // valid only if gi%P==rank
    return (gi - rank) / P;
}

static inline int num_local_rows_cyclic(int M, int rank, int P) {
    if (rank >= M) return 0;
    return 1 + (M - 1 - rank) / P;
}

// ============================================================
// MPI datatype for COOEntry and RowVal
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

static MPI_Datatype make_mpi_rowval_type()
{
    MPI_Datatype MPI_RV;
    RowVal d{};
    int bl[2] = {1,1};
    MPI_Aint disp[2], base;
    MPI_Get_address(&d, &base);
    MPI_Get_address(&d.row, &disp[0]);
    MPI_Get_address(&d.val, &disp[1]);
    disp[0] -= base; disp[1] -= base;
    MPI_Datatype types[2] = {MPI_INT32_T, MPI_DOUBLE};
    MPI_Type_create_struct(2, bl, disp, types, &MPI_RV);
    MPI_Type_commit(&MPI_RV);
    return MPI_RV;
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

    // g2l for ALL global columns
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

    // Preallocate per-iteration value buffers (no alloc in timing loop)
    plan.tmp_sendvals.assign(plan.recvcols.size(), 0.0);
    plan.tmp_recvvals.assign(plan.sendcols.size(), 0.0);
}

static inline void exchange_ghost_values(const GhostPlan& plan,
                                        const std::vector<int>& g2l,
                                        const std::vector<double>& x_local,
                                        std::vector<double>& x_local_rw)
{
    if (!plan.has_ghosts()) return;

    // send back values for columns others requested from me
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

    // receive values for my requested ghost columns (same order as sendcols)
    MPI_Alltoallv(plan.tmp_sendvals.data(), plan.recvcounts.data(), plan.rdispls.data(), MPI_DOUBLE,
                  plan.tmp_recvvals.data(), plan.sendcounts.data(), plan.sdispls.data(), MPI_DOUBLE,
                  MPI_COMM_WORLD);

    // write ghosts into x_local at precomputed local indices
    for (size_t k = 0; k < plan.tmp_recvvals.size(); ++k) {
        int lidx = plan.ghost_lidx[k];
        x_local_rw[(size_t)lidx] = plan.tmp_recvvals[k];
    }
}

// ============================================================
// VALIDATION (added back): reference y computed by rank0
// - reads matrix sequentially from file (rank0 only)
// - uses deterministic x_value(j)
// - gathers y from cyclic distribution using (row,val) pairs
// ============================================================

static bool validate_on_rank0(const std::string& mtxPath,
                              int M, int N,
                              int rank, int P,
                              const std::vector<double>& y_local,
                              ValidationResult& out)
{
    // Gather computed y into rank0 (cyclic rows) as (globalRow, val)
    int localM = (int)y_local.size();
    std::vector<RowVal> send(localM);
    for (int lr = 0; lr < localM; ++lr) {
        int gi = rank + lr * P;
        send[(size_t)lr] = RowVal{(int32_t)gi, y_local[(size_t)lr]};
    }

    std::vector<int> counts(P, 0), displs(P, 0);
    MPI_Gather(&localM, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<RowVal> all;
    int tot = 0;
    if (rank == 0) {
        for (int p = 0; p < P; ++p) { displs[p] = tot; tot += counts[p]; }
        all.resize((size_t)tot);
    }

    MPI_Datatype MPI_RV = make_mpi_rowval_type();
    MPI_Gatherv(send.data(), localM, MPI_RV,
                (rank==0? all.data(): nullptr),
                (rank==0? counts.data(): nullptr),
                (rank==0? displs.data(): nullptr),
                MPI_RV, 0, MPI_COMM_WORLD);
    MPI_Type_free(&MPI_RV);

    if (rank != 0) return true;

    std::vector<double> y_gather((size_t)M, 0.0);
    for (const auto& rv : all) {
        if (rv.row >= 0 && rv.row < M) y_gather[(size_t)rv.row] = rv.val;
    }

    // Build reference y by reading file sequentially (rank0 only)
    std::vector<double> y_ref((size_t)M, 0.0);

    std::ifstream fin(mtxPath);
    if (!fin) {
        std::cerr << "[validation] cannot open matrix for validation: " << mtxPath << "\n";
        return false;
    }

    std::string line;
    bool dims_seen = false;
    while (std::getline(fin, line)) {
        line = rtrim_cr(line);
        if (line.empty()) continue;
        if (line[0] == '%') continue;

        if (!dims_seen) {
            int m2=0,n2=0,nz2=0;
            if (!parse_dims_line(line, m2, n2, nz2)) continue;
            dims_seen = true;
            continue;
        }

        int i0, j0;
        double v;
        if (!parse_triplet_line(line, i0, j0, v)) continue;
        if (i0 < 0 || i0 >= M || j0 < 0 || j0 >= N) continue;

        y_ref[(size_t)i0] += v * x_value(j0);
    }

    // Compute error metrics
    long double num = 0.0L;
    long double den = 0.0L;
    long double max_abs = 0.0L;

    for (int i = 0; i < M; ++i) {
        long double diff = (long double)y_gather[(size_t)i] - (long double)y_ref[(size_t)i];
        long double ad   = std::fabsl(diff);
        if (ad > max_abs) max_abs = ad;
        num += diff * diff;
        den += (long double)y_ref[(size_t)i] * (long double)y_ref[(size_t)i];
    }

    out.max_abs_error = max_abs;
    out.rel_L2_error  = (den > 0.0L) ? std::sqrt(num / den) : std::sqrt(num);

    return true;
}

// ============================================================
// Logging (human readable + TSV) like D1
// Keeps "as before" + validation + robust results path.
// ============================================================

static void append_log_rank0(const char* argv0,
                            const std::string& mtx,
                            int M, int N,
                            long long nz_header,
                            long long nz_used,
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
                            double p90_e2e_ms,
                            double p90_comp_ms,
                            double p90_comm_ms,
                            double gflops_e2e,
                            double gbps_e2e,
                            double gflops_comp,
                            double gbps_comp,
                            double comm_kib_max,
                            double mem_mib_max)
{
    fs::path root = repo_root_from_exe(argv0);
    fs::path results_dir = root / "results";
    std::error_code ec;
    fs::create_directories(results_dir, ec);

    // Human-readable log
    try {
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
             << ", nnz(header) = " << nz_header
             << ", nnz(used) = "   << nz_used << ")\n";
        fout << "Config    : ranks=" << ranks
             << ", threads=" << threads
             << ", schedule=" << lower_copy(schedule)
             << "(chunk=" << chunk << ")"
             << ", warmup=" << warmup
             << ", repeats=" << repeats
             << ", trials=" << trials
             << ", sort_rows=" << sort_rows
             << ", validation=" << do_validation << "\n";
        fout << "I/O       : MPI-IO chunk parsing (redistributed by row owner)\n";
        fout << "Comm      : ghost exchange via Alltoallv of double values (each iteration)\n";
        fout << std::fixed << std::setprecision(3);
        fout << "Metrics   : commKiB_max=" << comm_kib_max
             << ", memMiB_max=" << mem_mib_max << "\n";

        if (do_validation) fout << "Validation: " << validation_line.str() << "\n";
        else              fout << "Validation: (skipped)\n";

        fout << std::fixed << std::setprecision(3);
        fout << "Results   : P90_e2e_ms=" << p90_e2e_ms
             << ", GFLOPS_e2e=" << gflops_e2e
             << ", GBps_e2e=" << gbps_e2e << "\n";
        fout << "           : P90_comp_ms=" << p90_comp_ms
             << ", GFLOPS_comp=" << gflops_comp
             << ", GBps_comp=" << gbps_comp << "\n";
        fout << "           : P90_comm_ms=" << p90_comm_ms << "\n";
        fout << "=================================================================\n\n";
    } catch (...) {
        // ignore
    }

    // TSV log
    try {
        fs::path tsv_path = results_dir / "spmv_mpi_results.tsv";
        bool write_header = !fs::exists(tsv_path);

        std::ofstream fout(tsv_path, std::ios::app);
        if (!fout) return;

        if (write_header) {
            fout << "timestamp\tmatrix\tM\tN\tnz_header\tnz_used\tranks\tthreads\tsched\tchunk\twarmup\trepeats\ttrials\tsort_rows\t"
                    "p90_e2e_ms\tp90_comp_ms\tp90_comm_ms\tgflops_e2e\tgbps_e2e\tgflops_comp\tgbps_comp\t"
                    "commKiB_max\tmemMiB_max\tvalidation\trelL2\tmaxAbs\n";
        }

        auto now   = std::chrono::system_clock::now();
        std::time_t t_c = std::chrono::system_clock::to_time_t(now);
        std::tm* tm = std::localtime(&t_c);
        std::ostringstream ts;
        ts << std::put_time(tm, "%Y-%m-%d_%H:%M:%S");

        fout << ts.str() << "\t"
             << mtx << "\t"
             << M << "\t" << N << "\t"
             << nz_header << "\t" << nz_used << "\t"
             << ranks << "\t" << threads << "\t"
             << lower_copy(schedule) << "\t" << chunk << "\t"
             << warmup << "\t" << repeats << "\t" << trials << "\t"
             << sort_rows << "\t"
             << std::setprecision(9) << p90_e2e_ms << "\t"
             << std::setprecision(9) << p90_comp_ms << "\t"
             << std::setprecision(9) << p90_comm_ms << "\t"
             << std::setprecision(9) << gflops_e2e << "\t"
             << std::setprecision(9) << gbps_e2e << "\t"
             << std::setprecision(9) << gflops_comp << "\t"
             << std::setprecision(9) << gbps_comp << "\t"
             << std::setprecision(9) << comm_kib_max << "\t"
             << std::setprecision(9) << mem_mib_max << "\t"
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

    // Resolve matrix path on rank0 then broadcast
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

    // OpenMP config
    omp_set_dynamic(0);
    omp_set_num_threads(threads_req);

    int chunk_eff = chunk_arg;
    omp_sched_t sched_eff = parse_omp_schedule(schedule_arg, chunk_eff);
    omp_set_schedule(sched_eff, chunk_eff);

    // Read matrix (MPI-IO) -> COO CHUNK
    int M=0, N=0, nz_header=0;
    std::vector<COOEntry> coo_local;
    parallel_read_matrix_market_mpiio(mtxResolved.c_str(), rank, P, M, N, nz_header, coo_local);

    // Redistribute by row owner
    redistribute_coo_by_row_owner(coo_local, rank, P);

    // Build CSR local
    CSRLocal A = coo_to_csr_cyclic_rows(coo_local, M, rank, P);
    if (sort_rows) sort_csr_rows_by_col(A);

    // nnz_used across ranks
    long long local_nnz = (long long)A.val.size();
    long long nnz_used  = 0;
    MPI_Allreduce(&local_nnz, &nnz_used, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    // Build x_local + ghost plan, then do one initial exchange
    std::vector<double> x_local;
    std::vector<int> g2l;
    GhostPlan plan;
    build_ghost_plan_and_x(rank, P, N, A.col, x_local, g2l, plan);
    // initial exchange (fills ghosts)
    exchange_ghost_values(plan, g2l, x_local, x_local);

    // Warmup: do BOTH comm + compute
    std::vector<double> y_local;
    for (int w=0; w<warmup; ++w) {
        exchange_ghost_values(plan, g2l, x_local, x_local);
        spmv_csr_local_omp_runtime(A, x_local, g2l, y_local);
    }

    // Timing samples: comm-only, compute-only, e2e (max over ranks)
    std::vector<double> samples_e2e_ms, samples_comp_ms, samples_comm_ms;
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
            exchange_ghost_values(plan, g2l, x_local, x_local);
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

    // Compute per-rank comm volume (KiB) for ghost exchange VALUES per iteration
    double local_comm_kib = 0.0;
    {
        // per-iteration bytes = doubles sent + doubles received
        // send = recvcols.size() values to others; recv = sendcols.size() values from owners
        double bytes = 8.0 * ( (double)plan.recvcols.size() + (double)plan.sendcols.size() );
        local_comm_kib = bytes / 1024.0;
    }
    double comm_kib_max = 0.0;
    MPI_Reduce(&local_comm_kib, &comm_kib_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Estimate memory footprint per rank (MiB)
    auto bytes_vec_int    = [](const std::vector<int>& v)->double    { return (double)v.size() * (double)sizeof(int); };
    auto bytes_vec_double = [](const std::vector<double>& v)->double { return (double)v.size() * (double)sizeof(double); };

    double local_mem_bytes =
        bytes_vec_int(A.rowptr) + bytes_vec_int(A.col) + bytes_vec_double(A.val) +
        bytes_vec_double(x_local) + bytes_vec_int(g2l) +
        bytes_vec_double(y_local) +
        bytes_vec_int(plan.sendcounts) + bytes_vec_int(plan.sdispls) + bytes_vec_int(plan.sendcols) + bytes_vec_int(plan.ghost_lidx) +
        bytes_vec_int(plan.recvcounts) + bytes_vec_int(plan.rdispls) + bytes_vec_int(plan.recvcols) +
        bytes_vec_double(plan.tmp_sendvals) + bytes_vec_double(plan.tmp_recvvals);

    double local_mem_mib = local_mem_bytes / (1024.0 * 1024.0);
    double mem_mib_max = 0.0;
    MPI_Reduce(&local_mem_mib, &mem_mib_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ============================================================
    // VALIDATION (added back)
    // ============================================================
    ValidationResult vres{};
    int validation_ok = 1;
    if (do_validation) {
        bool okv = validate_on_rank0(mtxResolved, M, N, rank, P, y_local, vres);
        int ok_int = okv ? 1 : 0;
        MPI_Bcast(&ok_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        validation_ok = ok_int;
    }

    // ============================================================
    // Perf numbers on rank0 (use nnz_used)
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

        // END-TO-END
        double gflops_e2e = gflops_from_ms(p90_e2e_ms);
        double gbps_e2e   = gbps_from_ms(p90_e2e_ms);

        // COMPUTE-ONLY
        double gflops_comp = gflops_from_ms(p90_comp_ms);
        double gbps_comp   = gbps_from_ms(p90_comp_ms);

        int used_threads = omp_used_threads();

        omp_sched_t sched_now;
        int chunk_now;
        omp_get_schedule(&sched_now, &chunk_now);

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
        std::cout << "  Comm                 : ghost exchange (Alltoallv) each iteration\n";
        std::cout << "  Time metric          : 90th percentile (P90) of max-rank time\n\n";

        // Lines parsed by scripts (keep stable!)
        std::cout << "Per-rank max (KiB) total=" << comm_kib_max << "\n";
        std::cout << "Per-rank max (MiB) total=" << mem_mib_max << "\n\n";

        if (do_validation) {
            if (validation_ok) {
                std::cout << std::scientific;
                std::cout << "VALIDATION\n";
                std::cout << "  rel_L2_error         : " << (double)vres.rel_L2_error << "\n";
                std::cout << "  max_abs_error        : " << (double)vres.max_abs_error << "\n\n";
                std::cout << std::fixed;
            } else {
                std::cout << "VALIDATION\n";
                std::cout << "  (failed to run validation)\n\n";
            }
        } else {
            std::cout << "[validation] skipped\n\n";
        }

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

        // Write results to repo_root/results (robust)
        append_log_rank0(argv[0], mtxResolved, M, N,
                         (long long)nz_header, nnz_used,
                         P, used_threads,
                         omp_sched_name(sched_now), chunk_now,
                         warmup, repeats, trials,
                         sort_rows, do_validation, vres,
                         p90_e2e_ms, p90_comp_ms, p90_comm_ms,
                         gflops_e2e, gbps_e2e,
                         gflops_comp, gbps_comp,
                         comm_kib_max, mem_mib_max);

        fs::path root = repo_root_from_exe(argv[0]);
        std::cout << "[Rank0] Results appended to " << (root / "results" / "spmv_mpi_results.txt") << "\n";
    }

    MPI_Finalize();
    return 0;
}