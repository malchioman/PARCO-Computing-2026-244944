// D2-code/spmv_mpi.cpp
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

// Find matrix path (rank0), then broadcast
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

    return "";
}

static std::string bcast_string_from_rank0(std::string s, int rank) {
    int len = (rank == 0) ? (int)s.size() : 0;
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) s.assign((size_t)len, '\0');
    if (len > 0) MPI_Bcast(s.data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
    return s;
}

static fs::path repo_root_from_exe(const char* argv0) {
    // robust on Linux cluster: prefer /proc/self/exe
    try {
#if defined(__linux__)
        fs::path p = fs::read_symlink("/proc/self/exe");
        fs::path exe = fs::canonical(p);
        return exe.parent_path().parent_path(); // .../bin/spmv_mpi -> repo root
#else
        fs::path exe = fs::canonical(fs::path(argv0));
        return exe.parent_path().parent_path();
#endif
    } catch (...) {
        // fallback: assume current working dir is repo root or inside it
        try {
            fs::path cwd = fs::current_path();
            if (fs::exists(cwd / "bin")) return cwd;
            if (fs::exists(cwd.parent_path() / "bin")) return cwd.parent_path();
        } catch (...) {}
        return fs::current_path();
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

static void mpi_file_read_at_big(MPI_File fh, MPI_Offset off,
                                 char* buf, MPI_Offset len)
{
    const MPI_Offset MAX = (MPI_Offset)std::numeric_limits<int>::max();
    MPI_Offset done = 0;
    while (done < len) {
        MPI_Offset chunk = std::min(MAX, len - done);
        MPI_Status st;
        // NOT collective -> avoids deadlock when len differs among ranks
        MPI_File_read_at(fh, off + done, buf + (size_t)done, (int)chunk, MPI_CHAR, &st);
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
// Parallel read MatrixMarket with MPI-IO chunk parsing
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

     mpi_file_read_at_big(fh, read_start, raw.data(), read_len);

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
    A.rowptr.assign((size_t)A.localM + 1, 0);

    for (const auto& e : coo_local) {
        int lr = local_row_of_global((int)e.i, rank, P);
        if (0 <= lr && lr < A.localM) A.rowptr[(size_t)lr + 1]++;
    }

    for (int r = 0; r < A.localM; ++r)
        A.rowptr[(size_t)r + 1] += A.rowptr[(size_t)r];

    const int local_nnz = A.rowptr[(size_t)A.localM];
    A.col.assign((size_t)local_nnz, 0);
    A.val.assign((size_t)local_nnz, 0.0);

    std::vector<int> cursor = A.rowptr;
    for (const auto& e : coo_local) {
        int lr = local_row_of_global((int)e.i, rank, P);
        if (lr < 0 || lr >= A.localM) continue;
        int pos = cursor[(size_t)lr]++;
        A.col[(size_t)pos] = (int)e.j;
        A.val[(size_t)pos] = e.v;
    }

    return A;
}

// Optional sort by column within each CSR row
static void sort_csr_rows_by_col(CSRLocal& A)
{
    for (int r = 0; r < A.localM; ++r) {
        int a = A.rowptr[(size_t)r];
        int b = A.rowptr[(size_t)r + 1];
        int len = b - a;
        if (len <= 1) continue;

        std::vector<int> idx((size_t)len);
        for (int k = 0; k < len; ++k) idx[(size_t)k] = a + k;

        std::sort(idx.begin(), idx.end(),
                  [&](int p, int q){ return A.col[(size_t)p] < A.col[(size_t)q]; });

        std::vector<int>    ctmp((size_t)len);
        std::vector<double> vtmp((size_t)len);
        for (int k = 0; k < len; ++k) {
            ctmp[(size_t)k] = A.col[(size_t)idx[(size_t)k]];
            vtmp[(size_t)k] = A.val[(size_t)idx[(size_t)k]];
        }
        for (int k = 0; k < len; ++k) {
            A.col[(size_t)a + (size_t)k] = ctmp[(size_t)k];
            A.val[(size_t)a + (size_t)k] = vtmp[(size_t)k];
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
        for (int k = A.rowptr[(size_t)r]; k < A.rowptr[(size_t)r + 1]; k++) {
            int gc  = A.col[(size_t)k];
            int idx = g2l[(size_t)gc];

#if SPMV_CHECK_G2L
            if (idx < 0 || idx >= (int)x_local.size()) {
                #pragma omp critical
                { std::cerr << "[fatal] missing/invalid g2l for col " << gc << "\n"; }
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
#endif
            sum += A.val[(size_t)k] * x_local[(size_t)idx];
        }
        y_local[(size_t)r] = sum;
    }
}

// ============================================================
// Percentile90 on rank0
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
    plan.sendcounts.assign((size_t)P, 0);
    for (int j : ghost_cols) plan.sendcounts[(size_t)(j % P)]++;

    plan.sdispls.assign((size_t)P, 0);
    for (int p = 1; p < P; ++p)
        plan.sdispls[(size_t)p] = plan.sdispls[(size_t)p - 1] + plan.sendcounts[(size_t)p - 1];

    plan.sendcols.resize((size_t)ghost_cols.size());
    {
        std::vector<int> cursor = plan.sdispls;
        for (int j : ghost_cols) {
            int owner = j % P;
            plan.sendcols[(size_t)cursor[(size_t)owner]++] = j;
        }
    }

    // Allocate ghost slots in x_local + fill ghost_lidx (same order as sendcols)
    plan.ghost_lidx.resize(plan.sendcols.size());
    for (size_t k = 0; k < plan.sendcols.size(); ++k) {
        int j = plan.sendcols[k];
        if (g2l[(size_t)j] == -1) {
            g2l[(size_t)j] = (int)x_local.size();
            x_local.push_back(0.0); // placeholder filled by exchange
        }
        plan.ghost_lidx[k] = g2l[(size_t)j];
    }

    // Exchange counts: recvcounts[p] = how many columns rank p asks from me
    plan.recvcounts.assign((size_t)P, 0);
    MPI_Alltoall(plan.sendcounts.data(), 1, MPI_INT,
                 plan.recvcounts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    plan.rdispls.assign((size_t)P, 0);
    int tot_recv = 0;
    for (int p = 0; p < P; ++p) {
        plan.rdispls[(size_t)p] = tot_recv;
        tot_recv += plan.recvcounts[(size_t)p];
    }
    plan.recvcols.resize((size_t)tot_recv);

    // Receive requested column IDs (others -> me)
    MPI_Alltoallv(plan.sendcols.data(), plan.sendcounts.data(), plan.sdispls.data(), MPI_INT,
                  plan.recvcols.data(), plan.recvcounts.data(), plan.rdispls.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // Preallocate value buffers for iterative exchange (NO alloc in timing loop)
    plan.tmp_sendvals.assign(plan.recvcols.size(), 0.0);
    plan.tmp_recvvals.assign(plan.sendcols.size(), 0.0);
}

// IMPORTANT: plan is NOT const (we write into tmp_* buffers)
static inline void exchange_ghost_values(GhostPlan& plan,
                                        const std::vector<int>& g2l,
                                        const std::vector<double>& x_local,
                                        std::vector<double>& x_work)
{
    if (!plan.has_ghosts()) return;

    // prepare sendvals: values for columns others requested from me
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

    // write ghosts into x_work at precomputed local indices
    for (size_t k = 0; k < plan.tmp_recvvals.size(); ++k) {
        int lidx = plan.ghost_lidx[k];
        x_work[(size_t)lidx] = plan.tmp_recvvals[k];
    }
}

// ============================================================
// Memory footprint (rough estimate of main vectors, bytes)
// ============================================================

static size_t bytes_of_vector_int(const std::vector<int>& v) {
    return v.size() * sizeof(int);
}
static size_t bytes_of_vector_double(const std::vector<double>& v) {
    return v.size() * sizeof(double);
}
static size_t bytes_of_vector_coo(const std::vector<COOEntry>& v) {
    return v.size() * sizeof(COOEntry);
}

static double bytes_to_mib(size_t b) {
    return (double)b / (1024.0 * 1024.0);
}

static double bytes_to_kib(size_t b) {
    return (double)b / 1024.0;
}

// ============================================================
// Validation
// ============================================================

struct CSRFull {
    int M=0, N=0;
    std::vector<int> rowptr;
    std::vector<int> col;
    std::vector<double> val;
};

static bool read_matrix_market_rank0_sequential(const std::string& path, CSRFull& Afull, long long& nz_header_out)
{
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;

    char buf[4096];
    bool got_dims = false;
    int M=0, N=0, nz=0;

    while (std::fgets(buf, (int)sizeof(buf), f)) {
        std::string line(buf);
        line = rtrim_cr(line);
        if (line.empty()) continue;
        if (!line.empty() && line[0] == '%') continue;
        if (parse_dims_line(line, M, N, nz)) {
            got_dims = true;
            break;
        }
    }

    if (!got_dims) { std::fclose(f); return false; }

    nz_header_out = (long long)nz;

    std::vector<COOEntry> coo;
    coo.reserve((size_t)std::max(1, nz));

    while (std::fgets(buf, (int)sizeof(buf), f)) {
        std::string line(buf);
        line = rtrim_cr(line);
        if (line.empty()) continue;
        if (!line.empty() && line[0] == '%') continue;

        int i0, j0;
        double v;
        if (!parse_triplet_line(line, i0, j0, v)) continue;
        if (i0 < 0 || i0 >= M || j0 < 0 || j0 >= N) continue;
        coo.push_back({(int32_t)i0, (int32_t)j0, v});
    }
    std::fclose(f);

    Afull.M = M;
    Afull.N = N;
    Afull.rowptr.assign((size_t)M + 1, 0);

    for (const auto& e : coo) Afull.rowptr[(size_t)e.i + 1]++;

    for (int r=0;r<M;r++) Afull.rowptr[(size_t)r+1] += Afull.rowptr[(size_t)r];

    const int nnz_used = Afull.rowptr[(size_t)M];
    Afull.col.assign((size_t)nnz_used, 0);
    Afull.val.assign((size_t)nnz_used, 0.0);

    std::vector<int> cursor = Afull.rowptr;
    for (const auto& e : coo) {
        int r = (int)e.i;
        int pos = cursor[(size_t)r]++;
        Afull.col[(size_t)pos] = (int)e.j;
        Afull.val[(size_t)pos] = e.v;
    }

    return true;
}

static void spmv_csr_full(const CSRFull& A, std::vector<double>& y)
{
    y.assign((size_t)A.M, 0.0);
    for (int r=0;r<A.M;r++) {
        double sum = 0.0;
        for (int k=A.rowptr[(size_t)r]; k<A.rowptr[(size_t)r+1]; k++) {
            int j = A.col[(size_t)k];
            sum += A.val[(size_t)k] * x_value(j);
        }
        y[(size_t)r] = sum;
    }
}

static bool validate_on_rank0(const std::string& mtxResolved,
                             int M, int N, int P,
                             const std::vector<double>& y_local,
                             ValidationResult& out,
                             bool force_validate,
                             long long nnz_used_global)
{
    const long long MAX_NNZ_VALIDATE = 5'000'000LL; // 5M entries
    const long long MAX_M_VALIDATE   = 2'000'000LL; // 2M rows

    if (!force_validate) {
        if (nnz_used_global > MAX_NNZ_VALIDATE || M > MAX_M_VALIDATE) {
            return false; // skipped
        }
    }

    std::vector<int> recvcounts(P,0), displs(P,0);
    for (int r=0;r<P;r++) recvcounts[r] = num_local_rows_cyclic(M, r, P);
    for (int r=1;r<P;r++) displs[r] = displs[r-1] + recvcounts[r-1];

    std::vector<double> y_gather((size_t)M, 0.0);

    MPI_Gatherv(y_local.data(), (int)y_local.size(), MPI_DOUBLE,
                y_gather.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    std::vector<double> y_global((size_t)M, 0.0);
    for (int r=0;r<P;r++) {
        int base = displs[r];
        int lm = recvcounts[r];
        for (int lr=0; lr<lm; lr++) {
            int gi = r + lr*P;
            if (gi < M) y_global[(size_t)gi] = y_gather[(size_t)base + (size_t)lr];
        }
    }

    CSRFull Afull;
    long long nz_header_ref = 0;
    if (!read_matrix_market_rank0_sequential(mtxResolved, Afull, nz_header_ref)) {
        std::cerr << "[validation] rank0 failed to read matrix sequentially\n";
        return false;
    }

    std::vector<double> y_ref;
    spmv_csr_full(Afull, y_ref);

    long double num = 0.0L;
    long double den = 0.0L;
    long double max_abs = 0.0L;

    for (int i=0;i<M;i++) {
        long double diff = (long double)y_global[(size_t)i] - (long double)y_ref[(size_t)i];
        long double ad   = (long double)std::fabs(diff);
        if (ad > max_abs) max_abs = ad;

        num += diff*diff;
        long double rv = (long double)y_ref[(size_t)i];
        den += rv*rv;
    }

    out.max_abs_error = max_abs;
    out.rel_L2_error  = (den > 0.0L) ? std::sqrt(num/den) : std::sqrt(num);

    return true;
}

// ============================================================
// Logging (always write under repo_root/results)
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
                            int validation_performed,
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
    fs::create_directories(results_dir);

    // Human-readable log
    {
        fs::path log_path = results_dir / "spmv_mpi_results.txt";
        bool write_header = !fs::exists(log_path);

        std::ofstream fout(log_path, std::ios::app);
        if (fout) {
            if (write_header) {
                fout << "SpMV MPI Benchmark - Run Log\n\n";
            }

            auto now   = std::chrono::system_clock::now();
            std::time_t t_c = std::chrono::system_clock::to_time_t(now);
            std::tm* tm = std::localtime(&t_c);

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
                 << ", validation=" << do_validation
                 << ", validation_performed=" << validation_performed << "\n";
            fout << std::fixed << std::setprecision(3);
            fout << "Times     : P90_e2e_ms="  << p90_e2e_ms
                 << ", P90_comp_ms=" << p90_comp_ms
                 << ", P90_comm_ms=" << p90_comm_ms << "\n";
            fout << "Perf      : GFLOPS_e2e="  << gflops_e2e
                 << ", GBps_e2e="    << gbps_e2e
                 << ", GFLOPS_comp=" << gflops_comp
                 << ", GBps_comp="   << gbps_comp << "\n";
            fout << "Comm/Iter : Per-rank max (KiB): total=" << comm_kib_max << "\n";
            fout << "Memory    : Per-rank max (MiB): total=" << mem_mib_max << "\n";

            if (do_validation && validation_performed) {
                fout.setf(std::ios::scientific, std::ios::floatfield);
                fout << "Validation: rel_L2_error=" << (long double)vres.rel_L2_error
                     << ", max_abs_error=" << (long double)vres.max_abs_error << "\n";
            } else if (do_validation && !validation_performed) {
                fout << "Validation: skipped (too large unless --validate-force)\n";
            } else {
                fout << "Validation: disabled (--no-validate)\n";
            }

            fout << "=================================================================\n\n";
        }
    }

    // TSV
    {
        fs::path tsv_path = results_dir / "spmv_mpi_results.tsv";
        bool write_header = !fs::exists(tsv_path);

        std::ofstream fout(tsv_path, std::ios::app);
        if (fout) {
            if (write_header) {
                fout << "timestamp\tmatrix\tM\tN\tnz_header\tnz_used\tranks\tthreads\tsched\tchunk\twarmup\trepeats\ttrials\tsort_rows\t"
                        "p90_e2e_ms\tp90_comp_ms\tp90_comm_ms\tgflops_e2e\tgbps_e2e\tgflops_comp\tgbps_comp\tcommKiB_max\tmemMiB_max\t"
                        "validation\tvalidation_performed\trelL2\tmaxAbs\n";
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
                 << validation_performed << "\t"
                 << (validation_performed ? (double)vres.rel_L2_error : 0.0) << "\t"
                 << (validation_performed ? (double)vres.max_abs_error : 0.0) << "\n";
        }
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
                      << " <matrix.mtx> <threads> [static|dynamic|guided|auto] [chunk] [repeats] [trials] "
                         "[--no-validate] [--validate-force] [--sort-rows]\n";
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
    int force_validate = 0;
    int sort_rows = 0;

    if (rank == 0) {
        for (int a=1;a<argc;a++){
            std::string s = argv[a];
            if (s == "--no-validate")      do_validation = 0;
            if (s == "--validate-force")   force_validate = 1;
            if (s == "--sort-rows")        sort_rows = 1;
        }
    }
    MPI_Bcast(&do_validation, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&force_validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sort_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resolve matrix path on rank0, then broadcast
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

    // Read matrix (MPI-IO) -> COO chunk, then redistribute by row owner
    int M=0, N=0, nz_header=0;
    std::vector<COOEntry> coo_local;
    parallel_read_matrix_market_mpiio(mtxResolved.c_str(), rank, P, M, N, nz_header, coo_local);

    redistribute_coo_by_row_owner(coo_local, rank, P);

    // Build CSR local
    CSRLocal A = coo_to_csr_cyclic_rows(coo_local, M, rank, P);
    if (sort_rows) sort_csr_rows_by_col(A);

    // Free COO to reduce memory footprint
    std::vector<COOEntry>().swap(coo_local);

    // nnz_used = sum actual nnz across ranks
    long long local_nnz = (long long)A.val.size();
    long long nnz_used  = 0;
    MPI_Allreduce(&local_nnz, &nnz_used, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    // Build x_local + ghost plan, and create a working x buffer with ghosts
    std::vector<double> x_owned_and_ghosts;
    std::vector<int> g2l;
    GhostPlan plan;
    build_ghost_plan_and_x(rank, P, N, A.col, x_owned_and_ghosts, g2l, plan);

    std::vector<double> x_work = x_owned_and_ghosts;
    exchange_ghost_values(plan, g2l, x_owned_and_ghosts, x_work);

    // Warmup
    std::vector<double> y_local;
    for (int w=0; w<warmup; ++w) {
        exchange_ghost_values(plan, g2l, x_owned_and_ghosts, x_work);
        spmv_csr_local_omp_runtime(A, x_work, g2l, y_local);
    }

    // Timing samples
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
            exchange_ghost_values(plan, g2l, x_owned_and_ghosts, x_work);
            double t1 = MPI_Wtime();
            spmv_csr_local_omp_runtime(A, x_work, g2l, y_local);
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

    // ============================================================
    // COMMUNICATION VOLUME PER ITERATION (ghost exchange only)
    //
    // - recvcols.size(): how many doubles I SEND (others request from me)
    // - sendcols.size(): how many doubles I RECEIVE (I request from others)
    //
    // total_per_rank = (send + recv) * sizeof(double)
    // ============================================================

    const unsigned long long local_comm_bytes_iter =
        (unsigned long long)(plan.sendcols.size() + plan.recvcols.size()) * (unsigned long long)sizeof(double);

    unsigned long long max_comm_bytes_iter = 0ULL;
    unsigned long long sum_comm_bytes_iter = 0ULL;

    MPI_Reduce(&local_comm_bytes_iter, &max_comm_bytes_iter, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_comm_bytes_iter, &sum_comm_bytes_iter, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Memory footprint estimate per rank (main vectors)
    size_t mem_bytes =
        bytes_of_vector_int(A.rowptr) +
        bytes_of_vector_int(A.col) +
        bytes_of_vector_double(A.val) +
        bytes_of_vector_double(x_owned_and_ghosts) +
        bytes_of_vector_double(x_work) +
        bytes_of_vector_int(g2l) +
        bytes_of_vector_int(plan.sendcounts) +
        bytes_of_vector_int(plan.sdispls) +
        bytes_of_vector_int(plan.sendcols) +
        bytes_of_vector_int(plan.ghost_lidx) +
        bytes_of_vector_int(plan.recvcounts) +
        bytes_of_vector_int(plan.rdispls) +
        bytes_of_vector_int(plan.recvcols) +
        bytes_of_vector_double(plan.tmp_sendvals) +
        bytes_of_vector_double(plan.tmp_recvvals) +
        bytes_of_vector_double(y_local);

    unsigned long long max_mem_bytes = 0ULL;
    unsigned long long sum_mem_bytes = 0ULL;
    {
        const unsigned long long local_mem = (unsigned long long)mem_bytes;
        MPI_Reduce(&local_mem, &max_mem_bytes, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_mem, &sum_mem_bytes, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    double commKiB_max = 0.0, commKiB_avg = 0.0;
    double memMiB_max = 0.0, memMiB_avg = 0.0;
    if (rank == 0) {
        commKiB_max = bytes_to_kib((size_t)max_comm_bytes_iter);
        commKiB_avg = bytes_to_kib((size_t)(sum_comm_bytes_iter / (unsigned long long)std::max(1, P)));

        memMiB_max = bytes_to_mib((size_t)max_mem_bytes);
        memMiB_avg = bytes_to_mib((size_t)(sum_mem_bytes / (unsigned long long)std::max(1, P)));
    }

    // Validation
    ValidationResult vres{};
    int validation_performed = 0;
    if (do_validation) {
        if (rank == 0) {
            bool did = validate_on_rank0(mtxResolved, M, N, P, y_local, vres,
                                         (force_validate != 0), nnz_used);
            validation_performed = did ? 1 : 0;
        }
        MPI_Bcast(&validation_performed, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&vres, sizeof(ValidationResult), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    // Perf numbers on rank0 using nnz_used
    if (rank == 0) {
        auto gflops_from_ms = [&](double ms)->double {
            return (ms > 0.0)
                ? (2.0 * (double)nnz_used) / (ms / 1000.0) / 1e9
                : std::numeric_limits<double>::infinity();
        };

        auto gbps_from_ms = [&](double ms)->double {
            double bytes = (double)nnz_used * (8.0 + 4.0 + 8.0) + (double)M * 8.0;
            return (ms > 0.0)
                ? bytes / (ms / 1000.0) / 1e9
                : std::numeric_limits<double>::infinity();
        };

        double gflops_e2e = gflops_from_ms(p90_e2e_ms);
        double gbps_e2e   = gbps_from_ms(p90_e2e_ms);

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
        std::cout << "  I/O                  : MPI-IO chunk parsing + redistribute by row owner\n";
        std::cout << "  Comm                 : ghost exchange (Alltoallv of doubles) each iteration\n";
        std::cout << "  Time metric          : 90th percentile (P90) of max-rank time\n\n";

        if (do_validation) {
            if (validation_performed) {
                std::cout.setf(std::ios::scientific, std::ios::floatfield);
                std::cout << "VALIDATION\n";
                std::cout << "  rel_L2_error         : " << (long double)vres.rel_L2_error << "\n";
                std::cout << "  max_abs_error        : " << (long double)vres.max_abs_error << "\n\n";
                std::cout.setf(std::ios::fixed, std::ios::floatfield);
            } else {
                std::cout << "VALIDATION\n";
                std::cout << "  status               : skipped (too large unless --validate-force)\n\n";
            }
        } else {
            std::cout << "VALIDATION\n";
            std::cout << "  status               : disabled (--no-validate)\n\n";
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

        // These two lines are parsed by your scripts (awk) -> DO NOT CHANGE WORDING
        std::cout << "COMM VOLUME PER ITERATION (ghost doubles only)\n";
        std::cout << "  Per-rank max (KiB): total=" << commKiB_max << "  avg=" << commKiB_avg << "\n";
        std::cout << "MEMORY FOOTPRINT (rough estimate)\n";
        std::cout << "  Per-rank max (MiB): total=" << memMiB_max << "  avg=" << memMiB_avg << "\n\n";

        std::cout << "=================================================================\n\n";

        // Always append results to repo_root/results
        append_log_rank0(argv[0], mtxResolved, M, N,
                         (long long)nz_header, nnz_used,
                         P, used_threads,
                         omp_sched_name(sched_now), chunk_now,
                         warmup, repeats, trials,
                         sort_rows,
                         do_validation, validation_performed, vres,
                         p90_e2e_ms, p90_comp_ms, p90_comm_ms,
                         gflops_e2e, gbps_e2e,
                         gflops_comp, gbps_comp,
                         commKiB_max, memMiB_max);

        fs::path root = repo_root_from_exe(argv[0]);
        std::cout << "[Rank0] Results appended to " << (root / "results" / "spmv_mpi_results.txt") << "\n";
    }

    MPI_Finalize();
    return 0;
}
