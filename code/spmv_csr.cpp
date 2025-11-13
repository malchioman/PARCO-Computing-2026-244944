
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <omp.h>

extern "C" {
  #include "mmio.h"
}

using namespace std;

struct CSR {
  int nrows = 0, ncols = 0;
  vector<int>   rowptr;
  vector<int>   col;
  vector<float> val;
};

static inline string lower_copy(string s){ for (auto& c:s) c=(char)tolower(c); return s; }

static inline bool file_exists(const string& p) {
  FILE* f = fopen(p.c_str(), "r");
  if (f) { fclose(f); return true; }
  return false;
}

CSR read_matrix_market_to_csr(const string& path){
  FILE* f = fopen(path.c_str(), "r");
  if (!f){ cerr<<"[fatal] cannot open "<<path<<"\n"; exit(1); }

  MM_typecode matcode;
  if (mm_read_banner(f, &matcode) != 0){ cerr<<"[fatal] mm_read_banner\n"; exit(1); }
  if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode)){
    cerr<<"[fatal] only 'matrix coordinate' files supported\n"; exit(1);
  }

  int M,N,nnz_file;
  if (mm_read_mtx_crd_size(f, &M, &N, &nnz_file) != 0){ cerr<<"[fatal] mm_read_mtx_crd_size\n"; exit(1); }

  vector<int> Ii; Ii.reserve(nnz_file * (mm_is_symmetric(matcode)? 2:1));
  vector<int> Jj; Jj.reserve(Ii.capacity());
  vector<float> Vv; Vv.reserve(Ii.capacity());

  for (int k=0;k<nnz_file;k++){
    int i,j; double v;
    if (mm_is_pattern(matcode)){
      if (fscanf(f, "%d %d", &i,&j) != 2){ cerr<<"[fatal] bad entry\n"; exit(1); }
      v = 1.0;
    } else if (mm_is_real(matcode) || mm_is_integer(matcode)){
      if (fscanf(f, "%d %d %lf", &i,&j,&v) != 3){ cerr<<"[fatal] bad entry\n"; exit(1); }
    } else { cerr<<"[fatal] unsupported data type (complex)\n"; exit(1); }

    Ii.push_back(i-1); Jj.push_back(j-1); Vv.push_back((float)v);
    if (mm_is_symmetric(matcode) && i!=j){
      Ii.push_back(j-1); Jj.push_back(i-1); Vv.push_back((float)v);
    }
  }
  fclose(f);

  const int nnz = (int)Vv.size();
  CSR A; A.nrows=M; A.ncols=N; A.rowptr.assign(M+1,0);

  for (int k=0;k<nnz;k++) A.rowptr[Ii[k]+1]++;
  for (int r=0;r<M;r++)   A.rowptr[r+1] += A.rowptr[r];

  A.col.assign(nnz,0); A.val.assign(nnz,0.0f);
  vector<int> next = A.rowptr;
  for (int k=0;k<nnz;k++){
    int r = Ii[k]; int p = next[r]++;
    A.col[p]=Jj[k]; A.val[p]=Vv[k];
  }
  return A;
}

vector<float> make_random_x(int n){
  mt19937 rng(42);
  normal_distribution<float> g(0.0f, 10.0f);
  vector<float> x(n);
  for (auto& e: x) e = g(rng);
  return x;
}

vector<float> matrixVectorMoltiplication(const CSR& A, const vector<float>& x,
                                         int threads, const string& schedule, int chunk)
{
  vector<float> y(A.nrows, 0.0f);
  omp_set_dynamic(0);
  omp_set_num_threads(max(1,threads));
  const string s = lower_copy(schedule);

  if (s=="dynamic"){
    #pragma omp parallel for schedule(dynamic,chunk)
    for (int i=0;i<A.nrows;i++){
      float acc=0.0f;
      for (int k=A.rowptr[i]; k<A.rowptr[i+1]; ++k)
        acc += A.val[k]*x[A.col[k]];
      y[i]=acc;
    }
  } else if (s=="guided"){
    #pragma omp parallel for schedule(guided,chunk)
    for (int i=0;i<A.nrows;i++){
      float acc=0.0f;
      for (int k=A.rowptr[i]; k<A.rowptr[i+1]; ++k)
        acc += A.val[k]*x[A.col[k]];
      y[i]=acc;
    }
  } else {
    #pragma omp parallel for schedule(static,chunk)
    for (int i=0;i<A.nrows;i++){
      float acc=0.0f;
      for (int k=A.rowptr[i]; k<A.rowptr[i+1]; ++k)
        acc += A.val[k]*x[A.col[k]];
      y[i]=acc;
    }
  }
  return y;
}

static inline double percentile90_ms(const CSR& A, const vector<float>& x,
                                     const string& schedule, int chunk, int threads,
                                     int warmup, int repeats, int trials)
{
  vector<float> y; y.reserve(A.nrows);

  for (int w=0; w<warmup; ++w)
    y = matrixVectorMoltiplication(A,x,threads,schedule,chunk);

  vector<double> samples; samples.reserve((size_t)repeats * (size_t)trials);

  for (int t=0; t<trials; ++t){
    for (int r=0; r<repeats; ++r){
      auto t0 = chrono::high_resolution_clock::now();
      y = matrixVectorMoltiplication(A,x,threads,schedule,chunk);
      auto t1 = chrono::high_resolution_clock::now();
      samples.push_back( chrono::duration<double, milli>(t1 - t0).count() );
    }
  }

  if (samples.empty()) return 0.0;

  size_t k = (size_t)ceil(0.90 * (double)samples.size());
  if (k == 0) k = 1;
  --k; // indice 0-based
  nth_element(samples.begin(), samples.begin()+k, samples.end());
  volatile float sink = (y.empty()? 0.0f : y[0]); (void)sink;
  return samples[k];
}

int main(int argc, char** argv){
  if (argc < 3){
    cerr << "Usage: " << argv[0]
         << " <matrix.mtx> <threads> [static|dynamic|guided] [chunk] [repeats] [trials]\n";
    return 1;
  }

  string mtxArg = argv[1];
  string mtx = mtxArg;

  if (!file_exists(mtx)) {
    const char* baseEnv = getenv("MATRICES_DIR");
    string base = baseEnv ? string(baseEnv) : string("matrices");
    string candidate = base + "/" + mtxArg;
    if (file_exists(candidate)) {
      mtx = candidate;
    } else {
      cerr << "[fatal] cannot open '" << mtxArg
           << "' nor '" << candidate << "'\n";
      return 1;
    }
  }

  const int    threads  = max(1, atoi(argv[2]));
  const string schedule = (argc>=4 ? string(argv[3]) : string("static"));
  const int    chunk    = (argc>=5 ? max(1, atoi(argv[4])) : 64);
  const int    repeats  = (argc>=6 ? max(1, atoi(argv[5])) : 10);
  const int    trials   = (argc>=7 ? max(1, atoi(argv[6])) : 5);
  const int    warmup   = 2;

  CSR A = read_matrix_market_to_csr(mtx);
  const long long nnz = (long long)A.val.size();
  cerr << "Loaded " << mtx << "  (" << A.nrows << "x" << A.ncols
       << ", nnz=" << nnz << ")\n";
  cerr << "measure: warmup=" << warmup << " repeats=" << repeats
       << " trials=" << trials << "  (report=P90)\n";

  vector<float> x = make_random_x(A.ncols);
  double p90_ms = percentile90_ms(A, x, schedule, chunk, threads, warmup, repeats, trials);

  double gflops = (p90_ms>0) ? (2.0*nnz)/(p90_ms/1000.0)/1e9 : INFINITY;
  double bytes  = nnz*(4.0+4.0+4.0) + A.nrows*4.0;
  double gbps   = (p90_ms>0) ? bytes/(p90_ms/1000.0)/1e9 : INFINITY;

  cout << fixed << setprecision(3)
       << "threads="  << threads
       << "  schedule=" << lower_copy(schedule) << "(" << chunk << ")"
       << "  p90_ms=" << p90_ms
       << "  GFLOPS="  << gflops
       << "  GBps="    << gbps
       << "\n";
  return 0;
}
