cmake_minimum_required(VERSION 3.15)
project(PARCO_Computing_2026_244944 LANGUAGES C CXX)

# ==== Base ====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ==== Sorgenti ====
set(CODE_DIR "${CMAKE_SOURCE_DIR}/code")

# Libreria C comune (mmio.c + mmio.h) per spmv_csr/custom_matrix
add_library(mmio STATIC "${CODE_DIR}/mmio.c")
target_include_directories(mmio PUBLIC "${CODE_DIR}")

# OpenMP per spmv_csr
find_package(OpenMP QUIET)

# ---- spmv_csr (C++ + OpenMP) ----
add_executable(spmv_csr "${CODE_DIR}/spmv_csr.cpp")
target_link_libraries(spmv_csr PRIVATE mmio)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(spmv_csr PRIVATE -O3 -march=native)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(spmv_csr PRIVATE OpenMP::OpenMP_CXX)
  else()
    target_compile_options(spmv_csr PRIVATE -fopenmp)
    target_link_options(spmv_csr PRIVATE -fopenmp)
  endif()
elseif (MSVC)
  target_compile_options(spmv_csr PRIVATE /O2 /openmp)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(spmv_csr PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

# ---- custom_matrix (ex gen_matrix) ----
add_executable(custom_matrix "${CODE_DIR}/custom_matrix.cpp")
target_link_libraries(custom_matrix PRIVATE mmio)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(custom_matrix PRIVATE -O2 -Wno-write-strings)
elseif (MSVC)
  target_compile_options(custom_matrix PRIVATE /O2)
endif()

# ---- make_matrices (nuovo tool; non usa mmio) ----
add_executable(make_matrices "${CODE_DIR}/make_matrices.cpp")
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(make_matrices PRIVATE -O2)
elseif (MSVC)
  target_compile_options(make_matrices PRIVATE /O2)
endif()

# ==== Install: metti tutti gli eseguibili in ./bin ====
install(TARGETS spmv_csr custom_matrix make_matrices RUNTIME DESTINATION bin)

# (opzionale) pacchetti
set(CPACK_GENERATOR "TGZ;ZIP")
set(CPACK_PACKAGE_NAME "parco")
set(CPACK_PACKAGE_VERSION "1.0.0")
include(CPack)
