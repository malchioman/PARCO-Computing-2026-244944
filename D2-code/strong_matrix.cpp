#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

static bool has_cmd(const std::string& cmd) {
#ifdef _WIN32
  std::string test = "where " + cmd + " >nul 2>nul";
#else
  std::string test = "command -v " + cmd + " >/dev/null 2>&1";
#endif
  return std::system(test.c_str()) == 0;
}

static int run_cmd(const std::string& cmd) {
  std::cout << "[fetch] cmd: " << cmd << "\n";
  return std::system(cmd.c_str());
}

static bool exists_nonempty(const fs::path& p) {
  std::error_code ec;
  return fs::exists(p, ec) && fs::is_regular_file(p, ec) && fs::file_size(p, ec) > 0;
}

static fs::path matrices_dir_from_exe(const char* argv0) {
  std::error_code ec;

  fs::path exePath = fs::path(argv0);
  if (exePath.is_relative()) {
    exePath = fs::current_path(ec) / exePath;
  }

  fs::path canon = fs::weakly_canonical(exePath, ec);
  if (!ec) exePath = canon;

  fs::path exeDir = exePath.parent_path();

  // Expect exe in repo_root/bin -> matrices in repo_root/matrices
  if (exeDir.filename() == "bin") {
    return (exeDir.parent_path() / "matrices").lexically_normal();
  }
  // Fallback
  return (exeDir / ".." / "matrices").lexically_normal();
}

static bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

int main(int argc, char** argv) {
  const std::string matrix = (argc >= 2) ? argv[1] : "kron_g500-logn21";
  if (matrix != "kron_g500-logn21") {
    std::cerr << "Per ora supporto solo: kron_g500-logn21\n";
    return 2;
  }

  if (!has_cmd("tar")) {
    std::cerr << "Errore: 'tar' non trovato nel PATH.\n";
    return 3;
  }
  if (!has_cmd("curl") && !has_cmd("wget")) {
    std::cerr << "Errore: serve 'curl' oppure 'wget' nel PATH.\n";
    return 4;
  }

  const fs::path dest_dir = matrices_dir_from_exe(argv[0]);
  fs::create_directories(dest_dir);

  const fs::path archive = dest_dir / (matrix + ".tar.gz");

  // Your cluster: HTTP works; HTTPS tends to reset
  const std::string url_http  = "http://sparse-files.engr.tamu.edu/MM/DIMACS10/" + matrix + ".tar.gz";
  const std::string url_https = "https://sparse-files.engr.tamu.edu/MM/DIMACS10/" + matrix + ".tar.gz";

  std::cout << "[fetch] matrix  = " << matrix << "\n";
  std::cout << "[fetch] exe     = " << argv[0] << "\n";
  std::cout << "[fetch] dest    = " << dest_dir.string() << "\n";
  std::cout << "[fetch] archive = " << archive.string() << "\n";
  std::cout << "[fetch] url     = " << url_http << "\n";

  // 1) Download (skip if already present)
  if (exists_nonempty(archive)) {
    std::cout << "[fetch] archive already exists, skip download.\n";
  } else {
    int rc = 1;

    if (has_cmd("curl")) {
      auto curl_download = [&](const std::string& url) -> int {
        // NOTE: do NOT use --http1.1 (not supported on your curl)
        // -C - enables resume (very useful on flaky/slow links)
        std::string cmd =
          "curl -L --fail "
          "--connect-timeout 15 "
          "--retry 10 --retry-all-errors --retry-delay 2 "
          "--speed-time 30 --speed-limit 1024 "
          "-C - "
          "-o \"" + archive.string() + "\" \"" + url + "\"";
        return run_cmd(cmd);
      };

      std::cout << "[fetch] downloading via curl (http)...\n";
      rc = curl_download(url_http);

      if (rc != 0) {
        std::cout << "[fetch] curl http failed (rc=" << rc << "), trying https...\n";
        rc = curl_download(url_https);
      }
    } else {
      auto wget_download = [&](const std::string& url) -> int {
        std::string cmd =
          "wget --tries=10 --timeout=15 -O \"" + archive.string() + "\" \"" + url + "\"";
        return run_cmd(cmd);
      };

      std::cout << "[fetch] downloading via wget (http)...\n";
      rc = wget_download(url_http);

      if (rc != 0) {
        std::cout << "[fetch] wget http failed (rc=" << rc << "), trying https...\n";
        rc = wget_download(url_https);
      }
    }

    if (rc != 0 || !exists_nonempty(archive)) {
      std::cerr << "[fetch] ERROR: download failed.\n";
      std::cerr << "[fetch] Try manually:\n";
      std::cerr << "  curl -L -C - -o " << archive.string() << " " << url_http << "\n";
      return 5;
    }

    std::cout << "[fetch] download OK.\n";
  }

  // 2) Extract
  std::cout << "[fetch] extracting...\n";
  std::string ex = "tar -xzf \"" + archive.string() + "\" -C \"" + dest_dir.string() + "\"";
  int exrc = run_cmd(ex);
  if (exrc != 0) {
    std::cerr << "[fetch] ERROR: extraction failed.\n";
    return 6;
  }

  // 3) Find a .mtx or .mtx.gz
  fs::path found;
  for (auto const& entry : fs::recursive_directory_iterator(dest_dir)) {
    if (!entry.is_regular_file()) continue;
    const fs::path p = entry.path();
    const std::string name = p.filename().string();
    if (ends_with(name, ".mtx") || ends_with(name, ".mtx.gz")) {
      found = p;
      break;
    }
  }

  if (!found.empty()) {
    std::cout << "[fetch] found matrix file: " << found.string() << "\n";
    // Move to matrices root for convenience
    if (found.parent_path() != dest_dir) {
      fs::path dst = dest_dir / found.filename();
      std::error_code ec;
      fs::rename(found, dst, ec);
      if (!ec) {
        std::cout << "[fetch] moved to: " << dst.string() << "\n";
      } else {
        std::cout << "[fetch] could not move (ok). Keeping at: " << found.string() << "\n";
      }
    }
  } else {
    std::cout << "[fetch] WARNING: no .mtx or .mtx.gz found under " << dest_dir.string() << "\n";
  }

  std::cout << "[fetch] done.\n";
  return 0;
}
