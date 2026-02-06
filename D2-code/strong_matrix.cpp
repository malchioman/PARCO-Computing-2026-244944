#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

static int run(const std::string& cmd) {
  std::cout << "[fetch] " << cmd << "\n";
  return std::system(cmd.c_str());
}

static bool exists_nonempty(const fs::path& p) {
  std::error_code ec;
  return fs::exists(p, ec) && fs::is_regular_file(p, ec) && fs::file_size(p, ec) > 0;
}

static fs::path matrices_dir_next_to_exe(const char* argv0) {
  std::error_code ec;
  fs::path exe = fs::path(argv0);
  if (exe.is_relative()) exe = fs::current_path(ec) / exe;
  exe = fs::weakly_canonical(exe, ec);
  return (exe.parent_path() / "matrices").lexically_normal(); // bin/matrices
}

int main(int argc, char** argv) {
  const std::string matrix = (argc >= 2) ? argv[1] : "kron_g500-logn21";
  if (matrix != "kron_g500-logn21") {
    std::cerr << "Supporto solo kron_g500-logn21\n";
    return 1;
  }

  const fs::path dest = matrices_dir_next_to_exe(argv[0]); // bin/matrices
  fs::create_directories(dest);

  const fs::path final_mtx = dest / (matrix + ".mtx");
  const fs::path archive   = dest / (matrix + ".tar.gz");

  // check if the matrix alredy exist
  if (exists_nonempty(final_mtx)) {
    std::cout << "[fetch] already have " << final_mtx.string() << "\n";
    return 0;
  }

  // HTTP url
  const std::string url =
    "http://sparse-files.engr.tamu.edu/MM/DIMACS10/kron_g500-logn21.tar.gz";

  // 1) Download (curl minimal, resume)
  if (!exists_nonempty(archive)) {
    int rc = run("curl -L -f --retry 10 -C - -o \"" + archive.string() + "\" \"" + url + "\"");
    if (rc != 0 || !exists_nonempty(archive)) {
      std::cerr << "[fetch] download failed\n";
      return 2;
    }
  } else {
    std::cout << "[fetch] archive already exists, skip download\n";
  }

  // 2) Extract into dest (creates dest/kron_g500-logn21/kron_g500-logn21.mtx)
  int exrc = run("tar -xzf \"" + archive.string() + "\" -C \"" + dest.string() + "\"");
  if (exrc != 0) {
    std::cerr << "[fetch] extraction failed\n";
    return 3;
  }

  // 3) Move Matrix Market to dest as kron_g500-logn21.mtx
  fs::path extracted_mtx = dest / matrix / (matrix + ".mtx"); // path standard in this tarball
  if (!exists_nonempty(extracted_mtx)) {
    std::cerr << "[fetch] expected file not found: " << extracted_mtx.string() << "\n";
    return 4;
  }

  std::error_code ec;
  fs::rename(extracted_mtx, final_mtx, ec);
  if (ec) {
    // fallback copy
    fs::copy_file(extracted_mtx, final_mtx, fs::copy_options::overwrite_existing, ec);
    if (ec) {
      std::cerr << "[fetch] move/copy failed\n";
      return 5;
    }
  }

  // 4) Cleanup: remove extracted folder and archive -> keep only .mtx
  fs::remove_all(dest / matrix, ec);
  fs::remove(archive, ec);

  std::cout << "[fetch] done: " << final_mtx.string() << "\n";
  return 0;
}
