cmake_minimum_required(VERSION 3.15)

project(PARCO_Computing_2026_244944 LANGUAGES C CXX)

# ==== Base settings ====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release if single-config (e.g. Makefiles, Ninja)
if(NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Directory with source files
set(CODE_DIR "${CMAKE_SOURCE_DIR}/code")

# ==== Output directories for binaries and libraries ====
# These are global defaults; we also reinforce them per-target below
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_SOURCE_DIR}/bin")

foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  string(TOUPPER ${cfg} CFG_UP)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG_UP} "${CMAKE_SOURCE_DIR}/bin")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CFG_UP} "${CMAKE_SOURCE_DIR}/bin")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CFG_UP} "${CMAKE_SOURCE_DIR}/bin")
endforeach()

# ==== Common C library (mmio.c + mmio.h) ====
add_library(mmio STATIC "${CODE_DIR}/mmio.c")
target_include_directories(mmio PUBLIC "${CODE_DIR}")

# ==== OpenMP (for spmv_csr) ====
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

# ---- custom_matrix ----
add_executable(custom_matrix "${CODE_DIR}/custom_matrix.cpp")
target_link_libraries(custom_matrix PRIVATE mmio)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(custom_matrix PRIVATE -O2 -Wno-write-strings)
elseif (MSVC)
  target_compile_options(custom_matrix PRIVATE /O2)
endif()

# ---- make_matrices ----
add_executable(make_matrices "${CODE_DIR}/make_matrices.cpp")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(make_matrices PRIVATE -O2)
elseif (MSVC)
  target_compile_options(make_matrices PRIVATE /O2)
endif()

# ==== Force output dir for all targets & all configs ====
# This is especially important for Visual Studio (Debug/Release/...)
foreach(tgt spmv_csr custom_matrix make_matrices)
  set_target_properties(${tgt} PROPERTIES
          RUNTIME_OUTPUT_DIRECTORY             "${CMAKE_SOURCE_DIR}/bin"
          RUNTIME_OUTPUT_DIRECTORY_DEBUG       "${CMAKE_SOURCE_DIR}/bin"
          RUNTIME_OUTPUT_DIRECTORY_RELEASE     "${CMAKE_SOURCE_DIR}/bin"
          RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_SOURCE_DIR}/bin"
          RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL  "${CMAKE_SOURCE_DIR}/bin"
  )
endforeach()

# ==== Optional install step (e.g. cmake --install) ====
install(TARGETS spmv_csr custom_matrix make_matrices
        RUNTIME DESTINATION bin)

# ==== Packaging (optional) ====
set(CPACK_GENERATOR "TGZ;ZIP")
set(CPACK_PACKAGE_NAME "parco")
set(CPACK_PACKAGE_VERSION "1.0.0")
include(CPack)
