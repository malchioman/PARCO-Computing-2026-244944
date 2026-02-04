#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <random>
#include <string>
#include <filesystem>

extern "C" {
  #include "../mmio/mmio.h"
}

namespace fs = std::filesystem;

// seed deterministico ma NON richiesto all'utente
static uint64_t hash64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

static fs::path resolve_output_path(const char* argv0, const std::string& outArg) {
  fs::path out = fs::path(outArg);

  // se l'utente passa solo un nome file, lo mettiamo in <exe_dir>/matrices/
  bool has_dir = out.has_parent_path() && out.parent_path() != fs::path(".");
  if (!has_dir) {
    fs::path exe_path = fs::absolute(fs::path(argv0));
    fs::path exe_dir  = exe_path.parent_path();
    fs::path matrices_dir = exe_dir / "matrices";
    fs::create_directories(matrices_dir);
    out = matrices_dir / out;
  } else {
    fs::create_directories(out.parent_path());
  }

  // forza estensione .mtx se manca
  if (out.extension() != ".mtx") out += ".mtx";
  return out;
}

int main(int argc, char** argv) {
  // Uso:
  //   ./bin/custom_matrix <out_path_or_name> <rows> <cols> <nnz>
  //
  // Esempio:
  //   ./bin/custom_matrix bin/matrices/weak_P4.mtx 65536 65536 4000000

  if (argc < 5) {
    std::fprintf(stderr,
      "Usage: %s <out_path_or_name> <rows> <cols> <nnz>\n"
      "Example: %s bin/matrices/weak_P4.mtx 65536 65536 4000000\n",
      argv[0], argv[0]);
    return 1;
  }

  const std::string outArg = argv[1];
  const int rows = std::max(1, std::atoi(argv[2]));
  const int cols = std::max(1, std::atoi(argv[3]));
  const long long nnz_ll = std::atoll(argv[4]);
  if (nnz_ll < 0) {
    std::fprintf(stderr, "[fatal] nnz must be >= 0\n");
    return 1;
  }
  const int nnz = (nnz_ll > INT32_MAX) ? INT32_MAX : (int)nnz_ll;

  fs::path out_path = resolve_output_path(argv[0], outArg);

  FILE* f = std::fopen(out_path.string().c_str(), "w");
  if (!f) {
    std::perror("[fatal] fopen");
    std::fprintf(stderr, "Path was: %s\n", out_path.string().c_str());
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

  // seed deterministico "automatico" (non lo inserisci tu)
  uint64_t seed = 12345ULL;
  seed ^= (uint64_t)rows * 0x9e3779b97f4a7c15ULL;
  seed ^= (uint64_t)cols * 0xbf58476d1ce4e5b9ULL;
  seed ^= (uint64_t)nnz  * 0x94d049bb133111ebULL;
  seed = hash64(seed);

  std::mt19937 rng((uint32_t)seed);
  std::uniform_int_distribution<int> Ri(1, rows);
  std::uniform_int_distribution<int> Cj(1, cols);
  std::uniform_real_distribution<double> Val(-10.0, 10.0);

  for (int k = 0; k < nnz; ++k) {
    std::fprintf(f, "%d %d %.6f\n", Ri(rng), Cj(rng), Val(rng));
  }

  std::fclose(f);

  // stampa 1 riga utile per debug/log
  std::printf("Matrix generated: %s (%dx%d, nnz=%d)\n",
              out_path.string().c_str(), rows, cols, nnz);

  return 0;
}
