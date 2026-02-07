


# Distributed Sparse Matrix–Vector Multiplication (SpMV) — OpenMP (D1) and MPI+OpenMP (D2)


## Introduction

This project contains the implementations used for the study:

**Distributed Sparse Matrix–Vector Multiplication (SpMV) — OpenMP (D1) and MPI+OpenMP (D2)**  
Author: **Massimo Malchiodi**, University of Trento

The repository provides two implementations of **Sparse Matrix–Vector Multiplication (SpMV)** on CPU:

- **D1 (OpenMP)**: shared-memory SpMV in **Compressed Sparse Row (CSR)** format, parallelized with a **row-wise loop decomposition** and configurable OpenMP scheduling (`static`, `dynamic`, `guided`).
- **D2 (MPI + OpenMP)**: hybrid distributed-memory + shared-memory SpMV using **MPI ranks** (distributed work) and optional **OpenMP threads** inside each rank.

The project includes:

- Main executables: `spmv_csr` (D1) and `spmv_mpi` (D2)
- Matrix tools: `make_matrices`, `custom_matrix`, `strong_matrix`
- Support for **Matrix Market (`.mtx`)** format through the NIST `mmio` library (`libmmio.a`)
- Helper scripts for benchmarking and scaling experiments (`script/`)

## Repository layout

The repository is organized as follows:

- `D1-code/` — Deliverable 1 sources (OpenMP CSR SpMV)
- `D2-code/` — Deliverable 2 sources (MPI + OpenMP hybrid SpMV)
- `mmio/` — Matrix Market I/O (NIST `mmio.c/h`, built as `libmmio.a`)
- `script/` — helper scripts (build, runs, scaling, PBS job files)
- `results/` — output logs produced by executables

After building, executables are placed in:

```
bin/
```


Input matrices (generated or downloaded) must be placed in:

```
bin/matrices/
```


### Executables produced in `bin/`

- `spmv_csr` — **D1** main executable (OpenMP CSR SpMV)
- `make_matrices` — generates default test matrices into `bin/matrices/`
- `custom_matrix` — generates a custom random matrix (user-defined shape and nnz)
- `spmv_mpi` — **D2** main executable (MPI + OpenMP hybrid SpMV)
- `strong_matrix` — downloads the real `.mtx` matrix used for **strong scaling** into `bin/matrices/`
- `libmmio.a` — static library used internally for Matrix Market I/O
---


## Requirements and Environment Setup

### Linux-first workflow

This repository (code + scripts) is designed to be built and run on **Linux** (including HPC clusters).
The D2 implementation requires an **MPI toolchain** (e.g., OpenMPI) and the provided scripts assume a Unix-like shell environment.

Native Windows is **not** the target workflow for this project.
Windows users should use **WSL2** (recommended) to get a consistent Linux toolchain.

---

### Windows users (WSL2)

If you are on Windows, use **WSL2** to build and run the project.

1) Install WSL2 (PowerShell as Administrator):
```powershell
wsl --install
```
Reboot if requested, then install Ubuntu from the Microsoft Store (if it is not installed automatically).
2) Inside Ubuntu (WSL terminal), install requirements:
```bash
sudo apt update
sudo apt install -y build-essential cmake git \
  openmpi-bin libopenmpi-dev
```

This provides:

- a C/C++ compiler toolchain (gcc/g++, make)

- CMake

- OpenMPI with the compiler wrappers mpicc and mpic++ (required by script/build.sh)

You can verify the setup with:
```bash
cmake --version
mpicc --version
mpirun --version
```
---
### Requirements
To build and run the project you need:

- CMake ≥ 3.15

- C++17 compiler with OpenMP support (e.g., GCC ≥ 9)

- MPI compiler wrappers available in PATH:
    1)  mpicc

    2) mpic++

Matrix Market I/O is provided via the included NIST mmio sources (built as libmmio.a).
___

### HPC / UniTN cluster environment (script/env.sh)

On the UniTN cluster the required toolchain is loaded via environment modules.
To set up a correct build/run environment, use:
```bash
source script/env.sh
```


What it does:

- initializes the module system

- loads:

  1) gcc91

  2) openmpi-3.0.0--gcc-9.1.0

  3) cmake-3.15.4

- exports OpenMPI compiler paths (OMPI_CC, OMPI_CXX) and a few runtime settings

- replaces the current shell with a fresh one that inherits the environment (exec bash)

After running it, you should see:
```
=== ENVIRONMENT READY ===
```


Note: env.sh is meant for the cluster module environment.
On a standard Linux/WSL installation you usually do not need it (just install packages and ensure mpicc/mpic++ exist).
___

### Clone the Repository

```bash
git clone https://github.com/malchioman/PARCO-Computing-2026-244944.git
cd PARCO-Computing-2026-244944
```
___ 
### Build (script/build.sh)

To simplify building (especially on the cluster), the repository provides:
```bash
script/build.sh   # CMake Release build using MPI compiler wrappers.
```

Run it from anywhere:
```bash 
bash script/build.sh
```


What it does:

- checks that mpicc and mpic++ are available in PATH
(if not, it stops with an error and suggests running script/env.sh)

- configures a Release build in build/

- builds with:

  1) DPARCO_BUILD_MPI=ON

  2) DPARCO_BUILD_OMP=ON

  3) DPARCO_MARCH_NATIVE=OFF

  4) CMAKE_C_COMPILER=mpicc

  5) CMAKE_CXX_COMPILER=mpic++

After a successful build, you should see:
```
=== BUILD COMPLETED ===
```


Executables are produced in:
```
bin/
```

#### Recommended cluster flow:
```bash
#from the root of the repository (the PARCO-Computing-2026-244944 directory)
source script/env.sh
script/build.sh
```
---


## Generate matrices (required before running SpMV)

All input matrices must be in **Matrix Market (`.mtx`)** format and are stored in:
```
bin/matrices/
```

### 1) Default benchmark matrices: `make_matrices`

`make_matrices` generates a set of **standard test matrices** and writes them into:

```
bin/matrices/
```

So to generate all standard matrices:

```bash
./bin/make_matrices
```

---
The following matrices are created by default:

| File Name           | Size (n × n)        | NNZ (approx.) | Type / Pattern                      | Description |
|---------------------|----------------------|----------------|--------------------------------------|-------------|
| **reg_150k.mtx**    | 150,000 × 150,000    | ~450,000       | 1-D Poisson (tridiagonal)            | Extremely regular, uniform row lengths |
| **irreg_50k.mtx**   | 50,000 × 50,000      | ~1,000,000     | Random sparse, fixed *k* nnz/row     | Irregular structure; good for dynamic/guided tests |
| **fem_5k.mtx**      | 5,154 × 5,154        | 99,199         | Random sparse, exact nnz             | FEM-like mesh sparsity pattern |
| **therm_1k.mtx**    | 1,228 × 1,228        | 8,598          | Random sparse, exact nnz             | Thermal simulation–style matrix |
| **rail_4k.mtx**     | 4,284 × 4,284        | 110,000        | Random sparse, exact nnz             | Medium irregular graph-like matrix |

All matrices use:

- 1-based indexing (Matrix Market convention)
- real, general, coordinate format
- internally built in CSR and exported to .mtx

make_matrices can also generate two optional, much larger matrices:

| File Name           | Size (n × n)         | NNZ        | Pattern                      | How to Enable |
|---------------------|-----------------------|------------|------------------------------|----------------|
| **social_280k.mtx** | 281,903 × 281,903     | 2,300,000  | Social-network-like graph    | `--social`     |
| **web_1M.mtx**      | 1,000,005 × 1,000,005 | 3,100,000  | Web-graph-style sparse graph | `--web`        |

using these commands:
```bash
./bin/make_matrices --social
./bin/make_matrices --web
```
---

#### Controlling the Irregularity of irreg_50k.mtx:

The matrix irreg_50k.mtx is generated using:
```bash
random_sparse_k(50000, k_irreg);
```

---

Where k_irreg is the number of nonzeros per row (default = 20).

You can modify this parameter with:
```bash
./bin/make_matrices --k 10     # very sparse
./bin/make_matrices --k 50     # denser
./bin/make_matrices --k 200    # heavy irregularity

```

---
Both forms are accepted:
```bash
./bin/make_matrices --k 35
./bin/make_matrices --k35
```

This is useful for stress-testing SpMV under different irregularity conditions.

--- 

### 2) Generate custom random matrices (`custom_matrix`)

custom_matrix generates a random Matrix Market file with a user-defined shape and number of nonzeros.

Usage:
```
./bin/custom_matrix <out_path_or_name> <rows> <cols> <nnz>
```

Examples:
```bash
# if you pass only a filename, it is written into bin/matrices/ automatically
./bin/custom_matrix weak_P4 65536 65536 4000000

# explicit output path (directories are created if needed)
./bin/custom_matrix bin/matrices/weak_P8.mtx 131072 131072 8000000
```

Notes:

- If the output argument has no directory, the file is saved into bin/matrices/.

- The .mtx extension is forced if missing.

- Entries are generated with random (i, j) (duplicates may appear) and values in [-10, +10].

- The RNG seed is deterministic (depends on rows/cols/nnz), so the same command reproduces the same matrix.
#### ⚠️ Generation may take longer and produce a large file.

---
### 3) Real-world matrix for strong scaling: strong_matrix

strong_matrix downloads a real .mtx matrix used for strong-scaling experiments and stores it into:
```
bin/matrices/
```
Run:
```bash
# from the root of the repository
./bin/strong_matrix
```

This downloads and produces:
```
bin/matrices/kron_g500-logn21.mtx
```
Notes:

- The tool currently supports only kron_g500-logn21 (it is the default; passing a different name exits with an error).

- It uses curl (with resume and retries) and tar under the hood.

- After extraction, it cleans up the .tar.gz and temporary folder, keeping only the final .mtx.

___

### Quick sanity check

After generation/download, you should see .mtx files here:
```bash
#from the root of the repository
ls -lah bin/matrices
```

___
## OpenMP CSR SpMV (Deliverable 1 (D1))

### What it does

Deliverable 1 implements **Sparse Matrix–Vector Multiplication** in **CSR (Compressed Sparse Row)** format on a single shared-memory machine.
The computation is parallelized with **OpenMP** using a **row-wise loop decomposition** and supports different scheduling policies:

- `static`
- `dynamic`
- `guided`

The code also performs a **numerical validation** by comparing the parallel result against a single-thread CSR reference implementation and reports:
- `rel_L2_err`  (relative L2 error)
- `max_abs_err` (maximum absolute error)

Results are logged automatically in:
```
results/spmv_results.txt
```

---

### Run the main executable (`spmv_csr`)

Syntax:

```bash
#from the root of the repository
./bin/spmv_csr <matrix.mtx> <threads> [static|dynamic|guided] [chunk] [repeats] [trials]
```


Parameters:

| Argument                | Description                                          |
| ----------------------- | ---------------------------------------------------- |
| `<matrix.mtx>`          | Input matrix in Matrix Market format                 |
| `<threads>`             | Number of OpenMP threads                             |
| `static/dynamic/guided` | Scheduling policy                                    |
| `[chunk]`               | Optional: OpenMP chunk size                          |
| `[repeats]`             | Optional: number of inner repetitions                |
| `[trials]`              | Optional: independent trials (P90 computed over all) |


Examples: 
```
# Static scheduling, 12 threads
./bin/spmv_csr bin/matrices/reg_150k.mtx 12 static
```
```
# Dynamic scheduling, chunk=64, 10 repeats, 5 trials
./bin/spmv_csr bin/matrices/irreg_50k.mtx 20 dynamic 64 10 5
```
```
# Guided scheduling
./bin/spmv_csr bin/matrices/irreg_50k.mtx 24 guided 128 5 3
```
___

### Output Logging (Automatic Result Saving)

Each execution of `spmv_csr` produces two complementary outputs:

1. **On-screen benchmark report**  
   A detailed human-readable summary is printed to the terminal at the end of each run.  
   It includes:
   - matrix information (size and number of nonzeros)
   - OpenMP configuration (threads, schedule, chunk size)
   - timing settings (warmup, repeats, trials)
   - validation metrics
   - P90 execution time, GFLOP/s, and bandwidth

2. **Automatic persistent logging**  
   Every run also appends a compact summary of the results to the file:
```
results/spmv_results.txt
```

The program automatically creates the `results/` directory and the log file if they do not exist.

Each entry in the log file contains a timestamp and a concise five-line summary, for example:
```
=================================================================
Run at 2025-11-22 15:42:10
Matrix : matrices/irreg_50k.mtx (50000 x 50000, nnz = 1000000)
Config : threads=20, schedule=dynamic(chunk=64), warmup=2, repeats=10, trials=5
Validation: rel_L2_error=1.23e-08, max_abs_error=3.45e-07
Results : p90_ms=0.123, GFLOPS=0.004, GBps=0.050
=================================================================
```
This log file makes it easy to track and compare multiple experiments without needing to copy terminal output manually. It is especially useful for instructors, reproducibility, or automated batch testing.

___ 

### helper scripts (script/)

The repository includes scripts to run reproducible D1 benchmarks on two predefined scenarios:

- Regular matrix → OpenMP static scheduling

- Irregular matrix → OpenMP dynamic scheduling

Scripts:
```
bash script/runreg.sh    # regular matrix run
bash script/runirr.sh    # irregular matrix run
bash script/seqreg.sh    # sequential run (regular)
bash script/seqirr.sh    # sequential run (irregular)
```


If scripts fail with “Permission denied”:
```
chmod +x script/*.sh
```
---

## MPI + OpenMP (hybrid SpMV) (Deliverable 2 (D2))

### What it does

Deliverable 2 implements **distributed Sparse Matrix–Vector Multiplication** using a hybrid model:

- **MPI ranks** distribute the work across processes
- **OpenMP threads** parallelize the local SpMV inside each rank

Key points of this implementation:

- **Input**: Matrix Market (`.mtx`)
- **I/O**: parallel **MPI-IO** reading (chunk-based), then redistribution of triplets
- **Distribution**: **1D cyclic row partitioning** (row `i` belongs to rank `i % P`)
- **Local format**: each rank builds a local CSR for its cyclic rows
- **Communication**: per-iteration **ghost/halo exchange** of required `x` entries via `MPI_Alltoallv`
- **Timing**: reports **P90** of the **max-rank** time across all iterations, split into:
    - end-to-end (comm + compute)
    - compute-only
    - comm-only

---

### Run the main executable (`spmv_mpi`)

Syntax (as printed by the program):

```bash
mpirun -np <ranks> ./bin/spmv_mpi <matrix.mtx> <threads> [static|dynamic|guided|auto] [chunk] [repeats] [trials] \
  [--no-validate] [--validate-force] [--sort-rows]
```

| Argument                | Description                                           |
|-------------------------|-------------------------------------------------------|
| `<ranks>`               | Number of MPI ranks (`mpirun -np`)                    |
| `<matrix.mtx>`          | Input matrix in Matrix Market format                  |
| `<threads>`             | Number of OpenMP threads per rank                     |
| `static/dynamic/guided` | OpenMP schedule (implemented via `schedule(runtime)`) |
| `[chunk]`               | OpenMP chunk size (default 64)                        |
| `[repeats]`             | inner repetitions per trial (default 10)              |
| `[trials]`              | number of trials (default 5)                          |
| `--no-validate`         | disable validation(optional)                          |
| `--validate-force`      | force validation even for large matrices (optional)   |
| `--sort-rows`           | sort CSR columns within each row (optional)           |

Examples:
```
# 4 MPI ranks, 2 threads each, dynamic schedule, chunk=64
mpirun -np 4 ./bin/spmv_mpi bin/matrices/kron_g500-logn21.mtx 2 dynamic 64 10 5
```
```
# 8 ranks, 1 thread, static schedule (defaults chunk=64 repeats=10 trials=5)
mpirun -np 8 ./bin/spmv_mpi bin/matrices/reg_150k.mtx 1 static
```
```
# Disable validation (useful for very large matrices)
mpirun -np 8 ./bin/spmv_mpi bin/matrices/kron_g500-logn21.mtx 2 dynamic 64 10 5 --no-validate
```
Matrix path resolution:

You can pass either a full path or just a filename.
If the file is not found as-is, rank 0 will try (in order):

- `matrices/<name>`

- `bin/matrices/<name>`

- `../bin/matrices/<name>`

- `$MATRICES_DIR/` (if `MATRICES_DIR` is set)
  
The resolved path is then broadcast to all ranks.

___
### Output Logging (Automatic Result Saving)

Each execution of spmv_mpi produces:

1) On-screen benchmark report (rank 0)
A detailed summary is printed to stdout including:

   - matrix info (M, N, nnz header vs nnz used)

   - MPI/OpenMP configuration (ranks, threads, schedule, chunk)

   - P90 timings (end-to-end / compute-only / comm-only)

   - GFLOPS and estimated bandwidth

   - communication volume per iteration (ghost exchange only)

   - rough memory footprint estimate per rank

2) Automatic persistent logging (rank 0)
Results are appended under the repository results/ directory (resolved from the executable path):

- Human-readable log:
```
results/spmv_mpi_results.txt
```
- TSV (easy to parse/plot):
```
results/spmv_mpi_results.tsv
```
___

### Numerical validation

Validation is **collective** (all ranks must participate):

- Each rank computes its local `y` (for its cyclic rows).
- Local `y` chunks are **gathered to rank 0**.
- Rank 0 reads the full matrix sequentially, computes a reference SpMV, and compares it against the gathered distributed result.

Reported metrics:

- `rel_L2_error` — relative L2 error ‖y_par − y_ref‖₂ / ‖y_ref‖₂
- `max_abs_error` — maximum absolute error maxᵢ |y_par[i] − y_ref[i]|

#### Automatic skip thresholds

To avoid excessive time/memory usage, validation is **automatically skipped** (unless forced) when:

- `nnz_used_global > 5,000,000`  (**more than 5 million nonzeros**), **or**
- `M > 2,000,000` (**more than 2 million rows**)

where `nnz_used_global` is the **actual** number of nonzeros loaded and used across all ranks (after parsing and redistribution).

#### Override flags

- `--validate-force` — force validation even if thresholds are exceeded
- `--no-validate` — disable validation entirely

Examples:

```bash
# Force validation (may be slow / memory heavy)
mpirun -np 8 ./bin/spmv_mpi bin/matrices/kron_g500-logn21.mtx 2 dynamic 64 10 5 --validate-force

# Disable validation
mpirun -np 8 ./bin/spmv_mpi bin/matrices/kron_g500-logn21.mtx 2 dynamic 64 10 5 --no-validate
```
___

### D2 helper scripts (`script/`)

The repository includes helper scripts to reproduce the **D2 experiments** used in the report:
- a generic MPI run (submitted via PBS)
- strong scaling
- weak scaling

All scripts assume:
- the executable is `bin/spmv_mpi`
- matrices are in `bin/matrices/`
- results are written under `results/`


> Note: some scripts parse specific lines from `spmv_mpi` output using `awk`.
> For this reason, a few print lines in `spmv_mpi` are intentionally kept stable.


---

####  `script/spmv_mpi.sh` (PBS submission wrapper)

This is the recommended way to run a **single MPI experiment on the cluster**.
It requires `qsub` and submits `script/spmv_mpi.pbs`.

Usage:

```bash
bash script/spmv_mpi.sh <NP> <threads> <matrix.mtx> [sched] [chunk] [repeats] [trials] [flags...]
```

Example:
```
script/spmv_mpi.sh 64 2 kron_g500-logn21.mtx dynamic 64 10 5 --no-validate
```

Important details:

- the matrix must exist in bin/matrices/ (the script uses the basename of <matrix.mtx>)

- defaults: QUEUE=short_cpuQ, WALLTIME=01:00:00, NCPUS_NODE=72

- the script automatically computes how many nodes to request and how many MPI ranks per node to place (RPN)

submission status is appended to a single master file:
```
results/pbs_out/spmv_mpi.out
```
___

####   `script/strong.sh` (strong scaling)

Runs a strong-scaling sweep: fixed matrix, varying the number of MPI ranks.

Usage:
```
bash script/strong.sh <matrix.mtx> [threads] [sched] [chunk] [repeats] [trials]
```

Default rank list inside the script:
```
P_LIST=(1 2 4 8 16 32 64 128)
```

Output:

a compact table is written to:
```
results/strong.txt
```

Notes:

the script sets `MATRICES_DIR=bin/matrices` and runs `mpirun -np P --bind-to none` ...

it extracts P90 times (e2e/compute/comm), GFLOPS/BW, and comm/memory estimates from the program output

___

####  `script/weak.sh` (weak scaling)

Runs a weak-scaling sweep: matrix size grows with the number of MPI ranks.

For each P in:
```
P_LIST=(1 2 4 8 16 32 64 128)
```

the script:

- generates (if missing) a matrix using bin/custom_matrix:
  - rows = ROWS_PER_RANK * P

  - cols = rows

  - nnz = NNZ_PER_RANK * P

- runs spmv_mpi with validation disabled (--no-validate)

- extracts metrics and appends them to a results table

Defaults (configurable via environment variables):

- ROWS_PER_RANK (default: 16384)

- NNZ_PER_RANK (default: 1000000)

Example:
```
export ROWS_PER_RANK=8192
export NNZ_PER_RANK=500000
bash script/weak.sh
```


Output:

a compact table is written to:
```
results/weak.txt
```
___

#### PBS templates (cluster)

- **Single-run submission**: `script/spmv_mpi.sh` submits the PBS job `script/spmv_mpi.pbs` via `qsub`
  (passing `NP`, `THREADS`, `MATRIX`, etc. as environment variables).

- **Strong/Weak scaling**: the PBS scripts are the entry points on the cluster:
    - `script/strong.pbs` loads modules and then calls `script/strong.sh`
    - `script/weak.pbs` loads modules, sets weak-scaling parameters, and then calls `script/weak.sh`

Examples (cluster):

```bash
qsub script/strong.pbs
qsub script/weak.pbs
```

Note:

when you use `qsub script/strong.pbs` make sure to have the matrix kron_g500-logn21.mtx in `bin/matrices`. If it is not present download it using `./bin/strong_matrix`

___
### Summary of executables

| Executable | Description |
|---|---|
| **spmv_csr** | **D1** main executable: CSR SpMV with configurable OpenMP scheduling (`static`, `dynamic`, `guided`) |
| **spmv_mpi** | **D2** main executable: distributed SpMV with **MPI ranks** + optional **OpenMP threads** per rank |
| **make_matrices** | Generates the default benchmark matrices into `bin/matrices/` (supports `--k`, `--social`, `--web`) |
| **custom_matrix** | Generates a custom random `.mtx` file: `custom_matrix <out> <rows> <cols> <nnz>` (output defaults to `bin/matrices/` if only a name is given) |
| **strong_matrix** | Downloads the real-world strong-scaling matrix (`kron_g500-logn21.mtx`) into `bin/matrices/` |
| **libmmio.a** | Static library for Matrix Market I/O (internal dependency, built from `mmio/`) |

---

## Notes

- Input matrices must be in **Matrix Market (`.mtx`)** format and stored in:
```
bin/matrices/
```

- For best performance, always compile in **Release** mode (recommended: ` bash script/build.sh`).

- Cluster runs:
  - use `source script/env.sh` to load the correct modules
  - for a single experiment use `bash script/spmv_mpi.sh ...` (submits `script/spmv_mpi.pbs`)
  - for scaling runs the recommended entry points are:
  ```bash
  qsub script/strong.pbs
  qsub script/weak.pbs
  ```

  - Strong scaling requires `kron_g500-logn21.mtx` in `bin/matrices/`.
  If it is missing, download it with:
```bash
./bin/strong_matrix
```
