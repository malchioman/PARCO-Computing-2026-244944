


# Parallel Sparse Matrix–Vector Multiplication in CSR Format (OpenMP)

## 1. Introduction

This project contains the implementation used for the study:

**Parallel Sparse Matrix–Vector Multiplication in CSR Format on Multicore CPUs with OpenMP**  
Author: **Massimo Malchiodi**, University of Trento

The repository provides a complete OpenMP-based implementation of **Sparse Matrix–Vector Multiplication (SpMV)** in **Compressed Sparse Row (CSR)** format.  
The computation is parallelized using a **row-wise loop decomposition** with configurable OpenMP scheduling (`static`, `dynamic`, `guided`).

The project includes:

- The main CSR SpMV executable (`spmv_csr`)
- Tools to generate matrices (`make_matrices`, `custom_matrix`)
- Support for the **Matrix Market (.mtx)** format through the NIST `mmio` library
- Scripts to run automated benchmarks on regular and irregular matrices

All executables are copied automatically into:

```

bin/

```

Input matrices (generated or downloaded) are stored in:

```

bin/matrices/

````

---

## 2. Requirements and Environment Setup

### 2.1 Requirements

To build and run the project you need:

- **CMake ≥ 3.15**
- **C++17 compiler** with OpenMP support  
  - GCC ≥ 9 (Linux)
  - Clang / Apple Clang (macOS)
  - MSVC or MinGW-w64 (Windows)
- Matrix Market I/O library (included as `mmio.c`)

---

### 2.2 Clone the Repository

```bash
git clone https://github.com/malchioman/PARCO-Computing-2026-244944.git
cd PARCO-Computing-2026-244944
````


---

### 2.3 Build Instructions

make sure to be in PARCO-Computing-2026-244944 directory
#### Linux / macOS

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

After building, executables will appear in:

```
bin/
```

#### Windows (Visual Studio)

```bat
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

Executables will be placed in:

```
bin\
```

---

### 2.4 Generate Matrices (Required Before Running SpMV)

#### 2.4.1 Generate standard test matrices

Run:

```bash
./bin/make_matrices
```

This creates a variety of regular and irregular matrices in:

```
bin/matrices/
```

---

#### 2.4.2 Generate custom random matrices (`custom_matrix`)

The `custom_matrix` executable generates a **random sparse matrix** with user-defined size and density.

Its logic is:

* creates directory `matrices/`
* sets default size = 10,000 × 10,000
* sets density = 0.5
* computes `nnz = rows × cols × density`
* writes a Matrix Market file with random:

    * row indices (1 to rows)
    * column indices (1 to cols)
    * values in [-10, +10]

The output file is always:

```
matrices/custom.mtx
```

**Run:**

```bash
cd bin
./custom_matrix
```

Result:

```
bin/matrices/large_random.mtx
```

> To change the matrix dimensions or density, edit the variables
> `rows`, `cols`, and `density` inside `custom_matrix.cpp`, then rebuild.

---

## 3. Scripts and Executables

### 3.1 Provided Scripts

The repository includes helper scripts to automatically execute benchmarks on two predefined scenarios:

* **Regular matrix → OpenMP `static` scheduling**
* **Irregular matrix → OpenMP `dynamic` scheduling**

These scripts are *optional* but useful for reproducible tests.

#### Linux / macOS

```bash
./runreg.sh    # Regular matrix, static scheduling
./runirr.sh    # Irregular matrix, dynamic scheduling
```

#### Windows (PowerShell)

```powershell
.\runreg.ps1
.\runirr.ps1
```

The scripts typically:

* loop over a fixed set of threads (1,2,4,8,...)
* choose an appropriate matrix
* execute `spmv_csr`
* store results in log files

---

### 3.2 Running the Main Executable (`spmv_csr`)

Use this executable to manually perform your own experiments.

Syntax:

```bash
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

#### Example commands

```bash
# Static scheduling, 12 threads
./bin/spmv_csr bin/matrices/irreg_50k.mtx 12 static
```

```bash
# Dynamic scheduling, chunk=64, 10 repeats, 5 trials
./bin/spmv_csr bin/matrices/irreg_50k.mtx 20 dynamic 64 10 5
```

```bash
# Guided scheduling on a custom-generated matrix
./bin/spmv_csr bin/matrices/large_random.mtx 24 guided 128 5 3
```

---

### 3.3 Summary of Executables

| Executable        | Description                                                                   |
| ----------------- | ----------------------------------------------------------------------------- |
| **spmv_csr**      | Runs the CSR SpMV kernel with configurable OpenMP scheduling                  |
| **make_matrices** | Generates default test matrices in `bin/matrices/`                            |
| **custom_matrix** | Generates a random matrix (`large_random.mtx`) with configurable size/density |
| **mmio**          | Utility for Matrix Market read/write (internal use)                           |
| **gen_matrix**    | Additional matrix generator (if enabled in CMake)                             |

---

## Notes

* All executables must be run **from the project root or from inside `bin/`**, depending on paths.
* Matrix Market files must reside in `bin/matrices/`.
* For best performance, always compile in `Release` mode.

---
