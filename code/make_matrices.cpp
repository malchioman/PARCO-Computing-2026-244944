
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <filesystem>
#ifdef _WIN32
#include <direct.h>
#else
#endif

struct CSR {
    int n=0, m=0;
    std::vector<int> row_ptr;   // n+1
    std::vector<int> col_idx;   // nnz
    std::vector<double> val;    // nnz
    size_t nnz() const { return val.size(); }
};

static std::mt19937_64& rng(){ static std::mt19937_64 g(42); return g; }

static void ensure_dir(const std::string& d){
#ifdef _WIN32
    _mkdir(d.c_str());
#else
    mkdir(d.c_str(), 0755);
#endif
}

static bool write_mtx(const std::string& path, const CSR& A){
    std::ofstream f(path);
    if(!f) { std::cerr<<"[x] errore scrittura "<<path<<"\n"; return false; }
    f << "%%MatrixMarket matrix coordinate real general\n";
    f << A.n << " " << A.m << " " << A.nnz() << "\n";
    f << std::scientific << std::setprecision(16);
    for(int i=0;i<A.n;i++)
        for(int k=A.row_ptr[i]; k<A.row_ptr[i+1]; ++k)
            f << (i+1) << " " << (A.col_idx[k]+1) << " " << A.val[k] << "\n";
    return true;
}

// --------- generator ----------
static CSR poisson1d(int n){
    CSR A; A.n=A.m=n;
    A.row_ptr.resize(n+1);
    A.col_idx.reserve(3LL*n-2);
    A.val.reserve(3LL*n-2);
    int p=0;
    for(int i=0;i<n;i++){
        A.row_ptr[i]=p;
        if(i>0){ A.col_idx.push_back(i-1); A.val.push_back(-1.0); ++p; }
        {       A.col_idx.push_back(i);   A.val.push_back( 2.0); ++p; }
        if(i<n-1){ A.col_idx.push_back(i+1); A.val.push_back(-1.0); ++p; }
    }
    A.row_ptr[n]=p;
    return A;
}

static CSR random_sparse_k(int n, int k, double diag_shift=1e-3){
    std::normal_distribution<double> N(0.0,1.0);
    std::uniform_int_distribution<int> U(0, n-1);
    std::vector<std::vector<std::pair<int,double>>> rows(n);
    for(int i=0;i<n;i++){
        std::set<int> used;
        for(int t=0;t<k; ++t){
            int j; do{ j = U(rng()); } while(used.count(j));
            used.insert(j);
            rows[i].push_back({j, N(rng())});
        }
        rows[i].push_back({i, diag_shift});
        std::sort(rows[i].begin(), rows[i].end());
    }
    CSR A; A.n=A.m=n; A.row_ptr.resize(n+1);
    size_t total=0;
    for(int i=0;i<n;i++){ total += rows[i].size(); A.row_ptr[i+1]=int(total); }
    A.col_idx.resize(total); A.val.resize(total);
    size_t p=0;
    for(int i=0;i<n;i++)
        for(auto &e : rows[i]){ A.col_idx[p]=e.first; A.val[p]=e.second; ++p; }
    return A;
}
static CSR random_sparse_exact_nnz(int n, int64_t nnz, bool force_diag=true){
    if(force_diag && nnz < n) nnz = n;
    std::uniform_int_distribution<int> U(0, n-1);
    std::set<std::pair<int,int>> S;
    if(force_diag) for(int i=0;i<n;i++) S.insert({i,i});
    while((int64_t)S.size() < nnz){
        int i = U(rng()), j = U(rng());
        S.insert({i,j});
    }
    CSR A; A.n=A.m=n; A.row_ptr.assign(n+1,0);
    A.col_idx.reserve(S.size()); A.val.reserve(S.size());
    int cur_i = 0;
    for(auto &e : S){
        int i=e.first, j=e.second;
        while(cur_i < i){ A.row_ptr[cur_i+1] = (int)A.col_idx.size(); ++cur_i; }
        A.col_idx.push_back(j);
        A.val.push_back(1.0); // pesi unitari (tipico per grafi)
    }
    while(cur_i < n){ A.row_ptr[cur_i+1] = (int)A.col_idx.size(); ++cur_i; }
    return A;
}

int main(int argc, char** argv){
    namespace fs = std::filesystem;

    bool add_social = false, add_web = false;
    int k_irreg = 20;

    for(int i=1;i<argc;i++){
        std::string s = argv[i];
        if(s == "--social") add_social = true;
        else if(s == "--web") add_web = true;
        else if(s == "--k" && i+1<argc) { k_irreg = std::stoi(argv[++i]); }
        else if(s.rfind("--k",0)==0){ k_irreg = std::stoi(s.substr(3)); }
    }

    fs::path exe_path = fs::absolute(fs::path(argv[0]));
    fs::path exe_dir  = exe_path.parent_path();

    fs::path matrices_dir = exe_dir / "matrices";
    ensure_dir(matrices_dir.string());

    auto mpath = [&](const char* name){
        return (matrices_dir / name).string();
    };

    {
        CSR A = poisson1d(150000);
        write_mtx(mpath("reg_150k.mtx"), A);
        std::cerr << "reg_150k.mtx created"<<"\n";
    }
    {
        CSR A = random_sparse_k(50000, k_irreg);
        write_mtx(mpath("irreg_50k.mtx"), A);
        std::cerr << "irreg_50k.mtx created" <<"\n";
    }
    {
        CSR A = random_sparse_exact_nnz(5154, 99199);
        write_mtx(mpath("fem_5k.mtx"), A);
        std::cerr << "fem_5k.mtx created"<<"\n";
    }
    {
        CSR A = random_sparse_exact_nnz(1228, 8598);
        write_mtx(mpath("therm_1k.mtx"), A);
        std::cerr << "therm_1k.mtx created"<<"\n";
    }
    {
        CSR A = random_sparse_exact_nnz(4284, 110000);
        write_mtx(mpath("rail_4k.mtx"), A);
        std::cerr << "rail_4k.mtxn created" <<"\n";
    }

    if(add_social){
        CSR A = random_sparse_exact_nnz(281903, 2300000);
        write_mtx(mpath("social_280k.mtx"), A);
        std::cerr << "social_280k.mtx created" << "\n";
    }
    if(add_web){
        CSR A = random_sparse_exact_nnz(1000005, 3100000);
        write_mtx(mpath("web_1M.mtx"), A);
        std::cerr << "web_1M.mtx created" <<"\n";
    }
    return 0;
}
