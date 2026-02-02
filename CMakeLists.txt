cmake_minimum_required(VERSION 3.15)

project(PARCO_Computing_2026_244944 LANGUAGES C CXX)

# ============================================================
# Base settings
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release if single-config (Makefiles, Ninja)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Build toggles
option(PARCO_BUILD_OMP "Build OpenMP targets (Deliverable 1)" ON)
option(PARCO_BUILD_MPI "Build MPI target (Deliverable 2)" ON)

# ============================================================
# Locate Deliverable 1 source dir robustly (code or D1-code)
# ============================================================
set(CODE_DIR_CANDIDATES
        "${CMAKE_SOURCE_DIR}/code"
        "${CMAKE_SOURCE_DIR}/D1-code"
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
          "  - ${CMAKE_SOURCE_DIR}/code\n"
          "  - ${CMAKE_SOURCE_DIR}/D1-code\n"
          "Fix: rename your D1 folder or update CODE_DIR_CANDIDATES."
  )
endif()

message(STATUS "Deliverable 1 CODE_DIR = ${CODE_DIR}")

# ============================================================
# Locate mmio dir (must be ./mmio)
# ============================================================
set(MMIO_DIR "${CMAKE_SOURCE_DIR}/mmio")
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
# Output directories (bin/)
# ============================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_SOURCE_DIR}/bin")

foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  string(TOUPPER ${cfg} CFG_UP)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG_UP} "${CMAKE_SOURCE_DIR}/bin")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CFG_UP} "${CMAKE_SOURCE_DIR}/bin")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CFG_UP} "${CMAKE_SOURCE_DIR}/bin")
endforeach()

# ============================================================
# Common C library (mmio)
# ============================================================
add_library(mmio STATIC "${MMIO_DIR}/mmio.c")
target_include_directories(mmio PUBLIC "${MMIO_DIR}")

# ============================================================
# Helper: enable OpenMP manually (robust on clusters)
# ============================================================
function(parco_enable_openmp tgt)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${tgt} PRIVATE -fopenmp)
    target_link_options(${tgt} PRIVATE -fopenmp)
  elseif (MSVC)
    # /openmp works for MSVC; keep it Release-only if you want
    target_compile_options(${tgt} PRIVATE /openmp)
  endif()
endfunction()

# ============================================================
# Helper: common release flags
# ============================================================
function(parco_release_flags tgt)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${tgt} PRIVATE -O3 -march=native)
  elseif (MSVC)
    target_compile_options(${tgt} PRIVATE $<$<CONFIG:Release>:/O2>)
  endif()
endfunction()

# ============================================================
# Deliverable 1 targets (OpenMP)
# ============================================================
add_executable(spmv_csr "${CODE_DIR}/spmv_csr.cpp")
target_link_libraries(spmv_csr PRIVATE mmio)
parco_release_flags(spmv_csr)

if (PARCO_BUILD_OMP)
  parco_enable_openmp(spmv_csr)
endif()

# ---- custom_matrix ----
if (EXISTS "${CODE_DIR}/custom_matrix.cpp")
  add_executable(custom_matrix "${CODE_DIR}/custom_matrix.cpp")
  target_link_libraries(custom_matrix PRIVATE mmio)
  parco_release_flags(custom_matrix)

  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(custom_matrix PRIVATE -Wno-write-strings)
  endif()
else()
  message(STATUS "custom_matrix.cpp not found in ${CODE_DIR}; target custom_matrix will be skipped.")
endif()

# ---- make_matrices ----
if (EXISTS "${CODE_DIR}/make_matrices.cpp")
  add_executable(make_matrices "${CODE_DIR}/make_matrices.cpp")
  parco_release_flags(make_matrices)
else()
  message(STATUS "make_matrices.cpp not found in ${CODE_DIR}; target make_matrices will be skipped.")
endif()

# ============================================================
# Deliverable 2 — MPI target (NO FindMPI, wrapper-based)
# Expected file at: D2-code/spmv_mpi.cpp
# ============================================================
if (PARCO_BUILD_MPI)
  set(D2_DIR "${CMAKE_SOURCE_DIR}/D2-code")
  set(D2_SRC "${D2_DIR}/spmv_mpi.cpp")

  if (EXISTS "${D2_SRC}")
    add_executable(spmv_mpi "${D2_SRC}")
    target_link_libraries(spmv_mpi PRIVATE mmio)
    parco_release_flags(spmv_mpi)

    # If later you make MPI+OpenMP hybrid, just enable this:
    # if (PARCO_BUILD_OMP)
    #   parco_enable_openmp(spmv_mpi)
    # endif()

    message(STATUS "D2 target enabled: spmv_mpi (wrapper-based MPI, no FindMPI)")
    message(STATUS "NOTE: build with mpicxx to get MPI includes/libs automatically.")
  else()
    message(STATUS "D2 source not found (${D2_SRC}); spmv_mpi target will be skipped.")
  endif()
endif()

# ============================================================
# Force output dir for all built targets & all configs
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

foreach(tgt IN LISTS _ALL_TARGETS)
  set_target_properties(${tgt} PROPERTIES
          RUNTIME_OUTPUT_DIRECTORY                 "${CMAKE_SOURCE_DIR}/bin"
          RUNTIME_OUTPUT_DIRECTORY_DEBUG           "${CMAKE_SOURCE_DIR}/bin"
          RUNTIME_OUTPUT_DIRECTORY_RELEASE         "${CMAKE_SOURCE_DIR}/bin"
          RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO  "${CMAKE_SOURCE_DIR}/bin"
          RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL      "${CMAKE_SOURCE_DIR}/bin"
  )
endforeach()

# ============================================================
# Install (optional)
# ============================================================
install(TARGETS ${_ALL_TARGETS} RUNTIME DESTINATION bin)

# ============================================================
# Packaging (optional)
# ============================================================
set(CPACK_GENERATOR "TGZ;ZIP")
set(CPACK_PACKAGE_NAME "parco")
set(CPACK_PACKAGE_VERSION "1.0.0")
include(CPack)
