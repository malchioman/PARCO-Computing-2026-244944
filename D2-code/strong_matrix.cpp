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

int main(int argc, char** argv) {
    const std::string matrix = (argc >= 2) ? argv[1] : "kron_g500-logn21";
    const fs::path dest_dir = (argc >= 3) ? fs::path(argv[2]) : fs::path("matrices");

    if (matrix != "kron_g500-logn21") {
        std::cerr << "Per ora supporto solo kron_g500-logn21.\n";
        return 2;
    }

    fs::create_directories(dest_dir);

    const std::string url =
        "https://sparse-files.engr.tamu.edu/MM/DIMACS10/" + matrix + ".tar.gz";
    const fs::path archive = dest_dir / (matrix + ".tar.gz");

    std::cout << "[fetch] matrix=" << matrix << "\n";
    std::cout << "[fetch] url=" << url << "\n";
    std::cout << "[fetch] dest=" << dest_dir.string() << "\n";

    std::string dl;
    if (has_cmd("curl")) {
        dl = "curl -L --fail --retry 3 --retry-delay 2 -o \"" + archive.string() + "\" \"" + url + "\"";
    } else if (has_cmd("wget")) {
        dl = "wget -O \"" + archive.string() + "\" \"" + url + "\"";
    } else {
        std::cerr << "Errore: serve curl o wget nel PATH.\n";
        return 3;
    }

    std::cout << "[fetch] downloading...\n";
    if (std::system(dl.c_str()) != 0) {
        std::cerr << "Errore download.\n";
        return 4;
    }

    if (!has_cmd("tar")) {
        std::cerr << "Errore: serve tar nel PATH per estrarre.\n";
        return 5;
    }

    std::string ex = "tar -xzf \"" + archive.string() + "\" -C \"" + dest_dir.string() + "\"";
    std::cout << "[fetch] extracting...\n";
    if (std::system(ex.c_str()) != 0) {
        std::cerr << "Errore estrazione.\n";
        return 6;
    }

    std::cout << "[fetch] done.\n";
    return 0;
}
