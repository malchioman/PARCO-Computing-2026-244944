cmake_minimum_required(VERSION 3.15)

project(PARCO_Computing_2026_244944 LANGUAGES C CXX)

# ============================================================
# Build toggles
# ============================================================
option(PARCO_BUILD_OMP "Build OpenMP targets (Deliverable 1)" ON)
option(PARCO_BUILD_MPI "Build MPI target (Deliverable 2)" ON)

# Su cluster: -march=native è rischioso (login != compute)
option(PARCO_MARCH_NATIVE "Enable -march=native (NOT recommended on clusters)" OFF)

# Dataset download targets (optional)
option(PARCO_FETCH_DATASETS "Enable dataset download targets" ON)

# Default Release (generator single-config)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
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
  )
endif()

message(STATUS "Deliverable 1 CODE_DIR = ${CODE_DIR}")

# ============================================================
# Locate mmio dir (must be ./mmio)
# ============================================================
set(MMIO_DIR "${CMAKE_SOURCE_DIR}/mmio")
if(NOT EXISTS "${MMIO_DIR}/mmio.c" OR NOT EXISTS "${MMIO_DIR}/mmio.h")
  message(FATAL_ERROR
          "mmio not found.\nExpected mmio/mmio.c and mmio/mmio.h at repo root."
  )
endif()

add_library(mmio STATIC "${MMIO_DIR}/mmio.c")
target_include_directories(mmio PUBLIC "${MMIO_DIR}")

# ============================================================
# Helpers: manual flags (no try_compile, no FindOpenMP/MPI)
# ============================================================
find_library(STDCXXFS_LIB stdc++fs)

function(parco_set_cpp17 tgt)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(${tgt} PRIVATE -std=gnu++17)
  elseif (MSVC)
    target_compile_options(${tgt} PRIVATE /std:c++17)
  endif()
endfunction()

function(parco_release_flags tgt)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(${tgt} PRIVATE $<$<CONFIG:Release>:-O3>)
    if (PARCO_MARCH_NATIVE)
      target_compile_options(${tgt} PRIVATE -march=native)
    endif()
  elseif (MSVC)
    target_compile_options(${tgt} PRIVATE $<$<CONFIG:Release>:/O2>)
  endif()
endfunction()

function(parco_enable_openmp_manual tgt)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(${tgt} PRIVATE -fopenmp)
    target_link_options(${tgt} PRIVATE -fopenmp)
  elseif (MSVC)
    target_compile_options(${tgt} PRIVATE /openmp)
  endif()
endfunction()

function(parco_maybe_link_stdcxxfs tgt)
  if (STDCXXFS_LIB)
    target_link_libraries(${tgt} PRIVATE ${STDCXXFS_LIB})
  endif()
endfunction()

function(parco_warnings tgt)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
  endif()
endfunction()

# ============================================================
# Deliverable 1 targets (OpenMP)
# ============================================================
add_executable(spmv_csr "${CODE_DIR}/spmv_csr.cpp")
target_link_libraries(spmv_csr PRIVATE mmio)
parco_set_cpp17(spmv_csr)
parco_release_flags(spmv_csr)
parco_warnings(spmv_csr)
parco_maybe_link_stdcxxfs(spmv_csr)

if (PARCO_BUILD_OMP)
  parco_enable_openmp_manual(spmv_csr)
endif()

# custom_matrix
if (EXISTS "${CODE_DIR}/custom_matrix.cpp")
  add_executable(custom_matrix "${CODE_DIR}/custom_matrix.cpp")
  target_link_libraries(custom_matrix PRIVATE mmio)
  parco_set_cpp17(custom_matrix)
  parco_release_flags(custom_matrix)
  parco_warnings(custom_matrix)
  parco_maybe_link_stdcxxfs(custom_matrix)

  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(custom_matrix PRIVATE -Wno-write-strings)
  endif()
else()
  message(STATUS "custom_matrix.cpp not found in ${CODE_DIR}; skipping.")
endif()

# make_matrices
if (EXISTS "${CODE_DIR}/make_matrices.cpp")
  add_executable(make_matrices "${CODE_DIR}/make_matrices.cpp")
  parco_set_cpp17(make_matrices)
  parco_release_flags(make_matrices)
  parco_warnings(make_matrices)
  parco_maybe_link_stdcxxfs(make_matrices)
else()
  message(STATUS "make_matrices.cpp not found in ${CODE_DIR}; skipping.")
endif()

# ============================================================
# Deliverable 2 target (MPI wrapper-based + OpenMP manual)
# IMPORTANT: configure with -DCMAKE_CXX_COMPILER=mpicxx / mpic++
# ============================================================
if (PARCO_BUILD_MPI)
  set(D2_SRC "${CMAKE_SOURCE_DIR}/D2-code/spmv_mpi.cpp")
  if (EXISTS "${D2_SRC}")
    add_executable(spmv_mpi "${D2_SRC}")
    parco_set_cpp17(spmv_mpi)
    parco_release_flags(spmv_mpi)
    parco_warnings(spmv_mpi)
    parco_maybe_link_stdcxxfs(spmv_mpi)

    # D2 includes omp.h and uses omp_*, so enable OpenMP flags too
    if (PARCO_BUILD_OMP)
      parco_enable_openmp_manual(spmv_mpi)
    endif()

    message(STATUS "D2 target enabled: spmv_mpi (MPI wrapper-based, OpenMP manual)")
    message(STATUS "IMPORTANT: configure with -DCMAKE_CXX_COMPILER=mpic++ (wrapper).")
  else()
    message(STATUS "D2 source not found (${D2_SRC}); spmv_mpi skipped.")
  endif()
endif()

# ============================================================
# Datasets (SuiteSparse Matrix Collection) - optional download
# CMake 3.15 compatible: extract all, then copy .mtx to matrices/
# ============================================================
set(PARCO_MATRICES_DIR "${CMAKE_SOURCE_DIR}/matrices" CACHE PATH "Directory for input matrices")
file(MAKE_DIRECTORY "${PARCO_MATRICES_DIR}")

if (PARCO_FETCH_DATASETS)
  find_program(CURL_EXECUTABLE curl)
  find_program(WGET_EXECUTABLE wget)

  function(parco_add_dataset target_name url out_file)
    set(OUT_PATH "${PARCO_MATRICES_DIR}/${out_file}")
    set(TAR_PATH "${PARCO_MATRICES_DIR}/${target_name}.tar.gz")

    set(EXTRACT_SCRIPT "${CMAKE_BINARY_DIR}/${target_name}_extract.cmake")
    file(WRITE "${EXTRACT_SCRIPT}" "
      if(NOT EXISTS \"${TAR_PATH}\")
        message(FATAL_ERROR \"Archive not found: ${TAR_PATH}\")
      endif()

      execute_process(
        COMMAND \"${CMAKE_COMMAND}\" -E tar xzf \"${TAR_PATH}\"
        WORKING_DIRECTORY \"${PARCO_MATRICES_DIR}\"
        RESULT_VARIABLE rc
      )
      if(NOT rc EQUAL 0)
        message(FATAL_ERROR \"Extraction failed (rc=\${rc})\")
      endif()

      file(GLOB_RECURSE hits
        \"${PARCO_MATRICES_DIR}/${out_file}\"
        \"${PARCO_MATRICES_DIR}/*/${out_file}\"
        \"${PARCO_MATRICES_DIR}/*/*/${out_file}\"
      )
      list(LENGTH hits n)
      if(n EQUAL 0)
        message(FATAL_ERROR \"Could not find ${out_file} after extraction in ${PARCO_MATRICES_DIR}\")
      endif()

      list(GET hits 0 src)
      message(STATUS \"Found ${out_file} at: \${src}\")
      execute_process(COMMAND \"${CMAKE_COMMAND}\" -E copy_if_different \"\${src}\" \"${OUT_PATH}\")
    ")

    if (CURL_EXECUTABLE)
      add_custom_target(${target_name}
              COMMAND ${CMAKE_COMMAND} -E echo "Downloading ${out_file} into ${PARCO_MATRICES_DIR}"
              COMMAND ${CURL_EXECUTABLE} -L -o "${TAR_PATH}" "${url}"
              COMMAND ${CMAKE_COMMAND} -P "${EXTRACT_SCRIPT}"
              COMMAND ${CMAKE_COMMAND} -E rm -f "${TAR_PATH}"
              BYPRODUCTS "${OUT_PATH}"
              COMMENT "Fetching dataset ${out_file}"
              VERBATIM
      )
    elseif (WGET_EXECUTABLE)
      add_custom_target(${target_name}
              COMMAND ${CMAKE_COMMAND} -E echo "Downloading ${out_file} into ${PARCO_MATRICES_DIR}"
              COMMAND ${WGET_EXECUTABLE} -O "${TAR_PATH}" "${url}"
              COMMAND ${CMAKE_COMMAND} -P "${EXTRACT_SCRIPT}"
              COMMAND ${CMAKE_COMMAND} -E rm -f "${TAR_PATH}"
              BYPRODUCTS "${OUT_PATH}"
              COMMENT "Fetching dataset ${out_file}"
              VERBATIM
      )
    else()
      message(WARNING
              "Neither 'curl' nor 'wget' found. Dataset targets will not work.\n"
              "Download matrices manually into: ${PARCO_MATRICES_DIR}"
      )
    endif()
  endfunction()

  # Strong scaling matrix (real): DIMACS10 / kron_g500-logn21
  parco_add_dataset(dataset_kron_g500_logn21
          "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn21.tar.gz"
          "kron_g500-logn21.mtx"
  )

  message(STATUS "Dataset target available: dataset_kron_g500_logn21")
  message(STATUS "Matrices directory: ${PARCO_MATRICES_DIR}")
endif()
