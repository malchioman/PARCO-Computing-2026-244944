cmake_minimum_required(VERSION 3.15)

project(PARCO_Computing_2026_244944 LANGUAGES C CXX)

# ============================================================
# Base settings
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if single-config generators (Makefiles, Ninja)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Build toggles
option(PARCO_BUILD_OMP "Build OpenMP targets (Deliverable 1)" ON)
option(PARCO_BUILD_MPI "Build MPI targets (Deliverable 2)" ON)

# On clusters: -march=native is risky (login node != compute nodes)
option(PARCO_MARCH_NATIVE "Enable -march=native (NOT recommended on clusters)" OFF)

# ============================================================
# FORCE output directories to <repo>/bin (robust with IDEs)
# ============================================================
set(PARCO_BIN_DIR "${PROJECT_SOURCE_DIR}/bin")

# Global defaults (some IDEs may override -> we also set per-target below)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PARCO_BIN_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PARCO_BIN_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PARCO_BIN_DIR}")

foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  string(TOUPPER ${cfg} CFG_UP)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG_UP} "${PARCO_BIN_DIR}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CFG_UP} "${PARCO_BIN_DIR}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CFG_UP} "${PARCO_BIN_DIR}")
endforeach()

function(parco_force_bin_output tgt)
  set_target_properties(${tgt} PROPERTIES
          RUNTIME_OUTPUT_DIRECTORY "${PARCO_BIN_DIR}"
          ARCHIVE_OUTPUT_DIRECTORY "${PARCO_BIN_DIR}"
          LIBRARY_OUTPUT_DIRECTORY "${PARCO_BIN_DIR}"
  )
  foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    string(TOUPPER ${cfg} CFG_UP)
    set_target_properties(${tgt} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_${CFG_UP} "${PARCO_BIN_DIR}"
            ARCHIVE_OUTPUT_DIRECTORY_${CFG_UP} "${PARCO_BIN_DIR}"
            LIBRARY_OUTPUT_DIRECTORY_${CFG_UP} "${PARCO_BIN_DIR}"
    )
  endforeach()
endfunction()

# ============================================================
# Locate Deliverable 1 source dir robustly (code or D1-code)
# ============================================================
set(CODE_DIR_CANDIDATES
        "${PROJECT_SOURCE_DIR}/code"
        "${PROJECT_SOURCE_DIR}/D1-code"
)

set(CODE_DIR "")
foreach(_cand IN LISTS CODE_DIR_CANDIDATES)
  if (EXISTS "${_cand}/spmv_csr.cpp")
    set(CODE_DIR "${_cand}")
    break()
  endif()
endforeach()

if (CODE_DIR STREQUAL "")
  message(FATAL_ERROR
          "Could not find Deliverable 1 sources.\n"
          "Expected spmv_csr.cpp inside one of:\n"
          "  - ${PROJECT_SOURCE_DIR}/code\n"
          "  - ${PROJECT_SOURCE_DIR}/D1-code\n"
          "Fix: rename your D1 folder or update CODE_DIR_CANDIDATES."
  )
endif()

message(STATUS "Deliverable 1 CODE_DIR = ${CODE_DIR}")

# ============================================================
# Locate mmio dir (must be ./mmio)
# ============================================================
set(MMIO_DIR "${PROJECT_SOURCE_DIR}/mmio")
if(NOT EXISTS "${MMIO_DIR}/mmio.c" OR NOT EXISTS "${MMIO_DIR}/mmio.h")
  message(FATAL_ERROR
          "mmio not found.\n"
          "Expected:\n"
          "  - ${MMIO_DIR}/mmio.c\n"
          "  - ${MMIO_DIR}/mmio.h\n"
          "Fix: ensure mmio.c and mmio.h are in the 'mmio' folder at repo root."
  )
endif()

# ============================================================
# Common C library (mmio)
# ============================================================
add_library(mmio STATIC "${MMIO_DIR}/mmio.c")
target_include_directories(mmio PUBLIC "${MMIO_DIR}")
parco_force_bin_output(mmio) # harmless; keeps archive in bin too

# ============================================================
# Packages (OpenMP/MPI)
# ============================================================
if(PARCO_BUILD_OMP)
  find_package(OpenMP REQUIRED)
endif()

if(PARCO_BUILD_MPI)
  find_package(MPI REQUIRED)
endif()

# ============================================================
# Helper: common release flags
# ============================================================
function(parco_release_flags tgt)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(${tgt} PRIVATE
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:RelWithDebInfo>:-O3>
    )
    if (PARCO_MARCH_NATIVE)
      target_compile_options(${tgt} PRIVATE -march=native)
    endif()
  elseif (MSVC)
    target_compile_options(${tgt} PRIVATE $<$<CONFIG:Release>:/O2>)
  endif()
endfunction()

# ============================================================
# Helper: enable OpenMP via imported target if available
# ============================================================
function(parco_link_openmp tgt)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
  else()
    message(FATAL_ERROR "OpenMP requested but not found by CMake.")
  endif()
endfunction()

# ============================================================
# Helper: filesystem workaround (GCC < 9 may need stdc++fs)
# ============================================================
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
  #include <filesystem>
  int main(){ std::filesystem::path p{\".\"}; return (int)p.string().size(); }
" PARCO_HAS_STD_FILESYSTEM)

function(parco_fix_filesystem tgt)
  if(NOT PARCO_HAS_STD_FILESYSTEM)
    target_link_libraries(${tgt} PRIVATE stdc++fs)
  endif()
endfunction()

# ============================================================
# Deliverable 1 targets
# ============================================================
add_executable(spmv_csr "${CODE_DIR}/spmv_csr.cpp")
parco_force_bin_output(spmv_csr)
target_link_libraries(spmv_csr PRIVATE mmio)
parco_release_flags(spmv_csr)
parco_fix_filesystem(spmv_csr)
if(PARCO_BUILD_OMP)
  parco_link_openmp(spmv_csr)
endif()

# ---- custom_matrix ----
if (EXISTS "${CODE_DIR}/custom_matrix.cpp")
  add_executable(custom_matrix "${CODE_DIR}/custom_matrix.cpp")
  parco_force_bin_output(custom_matrix)
  target_link_libraries(custom_matrix PRIVATE mmio)
  parco_release_flags(custom_matrix)
  parco_fix_filesystem(custom_matrix)

  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(custom_matrix PRIVATE -Wno-write-strings)
  endif()
else()
  message(STATUS "custom_matrix.cpp not found in ${CODE_DIR}; target custom_matrix will be skipped.")
endif()

# ---- make_matrices ----
if (EXISTS "${CODE_DIR}/make_matrices.cpp")
  add_executable(make_matrices "${CODE_DIR}/make_matrices.cpp")
  parco_force_bin_output(make_matrices)
  parco_release_flags(make_matrices)
  parco_fix_filesystem(make_matrices)
else()
  message(STATUS "make_matrices.cpp not found in ${CODE_DIR}; target make_matrices will be skipped.")
endif()

# ============================================================
# Deliverable 2 targets (MPI)
#   - spmv_mpi:      D2-code/spmv_mpi.cpp
#   - strong_matrix: D2-code/strong_matrix.cpp
# ============================================================
if(PARCO_BUILD_MPI)
  set(D2_DIR "${PROJECT_SOURCE_DIR}/D2-code")
  set(D2_SPMV_MPI_SRC      "${D2_DIR}/spmv_mpi.cpp")
  set(D2_STRONG_MATRIX_SRC "${D2_DIR}/strong_matrix.cpp")

  # ---- spmv_mpi ----
  if (EXISTS "${D2_SPMV_MPI_SRC}")
    add_executable(spmv_mpi "${D2_SPMV_MPI_SRC}")
    parco_force_bin_output(spmv_mpi)
    target_link_libraries(spmv_mpi PRIVATE MPI::MPI_CXX mmio)

    # If your spmv_mpi.cpp uses omp.h / omp calls, keep this ON
    if(PARCO_BUILD_OMP)
      parco_link_openmp(spmv_mpi)
    endif()

    parco_release_flags(spmv_mpi)
    parco_fix_filesystem(spmv_mpi)

    message(STATUS "D2 target enabled: spmv_mpi")
  else()
    message(STATUS "D2 source not found (${D2_SPMV_MPI_SRC}); spmv_mpi target will be skipped.")
  endif()

  # ---- strong_matrix ----
  if (EXISTS "${D2_STRONG_MATRIX_SRC}")
    add_executable(strong_matrix "${D2_STRONG_MATRIX_SRC}")
    parco_force_bin_output(strong_matrix)
    target_link_libraries(strong_matrix PRIVATE MPI::MPI_CXX mmio)

    # enable if strong_matrix.cpp uses omp.h
    if(PARCO_BUILD_OMP)
      parco_link_openmp(strong_matrix)
    endif()

    parco_release_flags(strong_matrix)
    parco_fix_filesystem(strong_matrix)

    message(STATUS "D2 target enabled: strong_matrix")
  else()
    message(STATUS "D2 source not found (${D2_STRONG_MATRIX_SRC}); strong_matrix target will be skipped.")
  endif()
endif()

# ============================================================
# Install (optional)
# ============================================================
set(_ALL_TARGETS spmv_csr)
if (TARGET custom_matrix)
  list(APPEND _ALL_TARGETS custom_matrix)
endif()
if (TARGET make_matrices)
  list(APPEND _ALL_TARGETS make_matrices)
endif()
if (TARGET spmv_mpi)
  list(APPEND _ALL_TARGETS spmv_mpi)
endif()
if (TARGET strong_matrix)
  list(APPEND _ALL_TARGETS strong_matrix)
endif()

install(TARGETS ${_ALL_TARGETS} RUNTIME DESTINATION bin)

# ============================================================
# Packaging (optional)
# ============================================================
set(CPACK_GENERATOR "TGZ;ZIP")
set(CPACK_PACKAGE_NAME "parco")
set(CPACK_PACKAGE_VERSION "1.0.0")
include(CPack)
