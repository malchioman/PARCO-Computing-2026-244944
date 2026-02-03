# ============================================================
# Datasets (SuiteSparse Matrix Collection) - optional download
# ============================================================
option(PARCO_FETCH_DATASETS "Enable dataset download targets" ON)

set(PARCO_MATRICES_DIR "${CMAKE_SOURCE_DIR}/matrices" CACHE PATH "Directory for input matrices")
file(MAKE_DIRECTORY "${PARCO_MATRICES_DIR}")

if (PARCO_FETCH_DATASETS)
  find_program(CURL_EXECUTABLE curl)
  find_program(WGET_EXECUTABLE wget)

  function(parco_add_dataset target_name url out_file)
    set(OUT_PATH "${PARCO_MATRICES_DIR}/${out_file}")
    set(TAR_PATH "${PARCO_MATRICES_DIR}/${target_name}.tar.gz")

    if (CURL_EXECUTABLE)
      add_custom_target(${target_name}
              COMMAND ${CMAKE_COMMAND} -E echo "Downloading ${out_file} into ${PARCO_MATRICES_DIR}"
              COMMAND ${CURL_EXECUTABLE} -L -o "${TAR_PATH}" "${url}"
              COMMAND ${CMAKE_COMMAND} -E tar xzf "${TAR_PATH}"
              COMMAND ${CMAKE_COMMAND} -E rm -f "${TAR_PATH}"
              WORKING_DIRECTORY "${PARCO_MATRICES_DIR}"
              BYPRODUCTS "${OUT_PATH}"
              COMMENT "Fetching dataset ${out_file}"
              VERBATIM
      )
    elseif (WGET_EXECUTABLE)
      add_custom_target(${target_name}
              COMMAND ${CMAKE_COMMAND} -E echo "Downloading ${out_file} into ${PARCO_MATRICES_DIR}"
              COMMAND ${WGET_EXECUTABLE} -O "${TAR_PATH}" "${url}"
              COMMAND ${CMAKE_COMMAND} -E tar xzf "${TAR_PATH}"
              COMMAND ${CMAKE_COMMAND} -E rm -f "${TAR_PATH}"
              WORKING_DIRECTORY "${PARCO_MATRICES_DIR}"
              BYPRODUCTS "${OUT_PATH}"
              COMMENT "Fetching dataset ${out_file}"
              VERBATIM
      )
    else()
      message(WARNING
              "Neither 'curl' nor 'wget' found. Dataset targets will not work.\n"
              "Install curl/wget or download matrices manually into: ${PARCO_MATRICES_DIR}"
      )
    endif()
  endfunction()

  # Strong scaling matrix (real): DIMACS10 / kron_g500-logn21 (Matrix Market tarball)
  # Source: SuiteSparse Matrix Collection mirror (MM tar.gz)
  parco_add_dataset(dataset_kron_g500_logn21
          "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn21.tar.gz"
          "kron_g500-logn21.mtx"
  )

  message(STATUS "Dataset target available: dataset_kron_g500_logn21")
  message(STATUS "Matrices directory: ${PARCO_MATRICES_DIR}")
endif()
