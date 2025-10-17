#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "roc::rocshmem" for configuration "Release"
set_property(TARGET roc::rocshmem APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(roc::rocshmem PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/librocshmem.a"
  )

list(APPEND _cmake_import_check_targets roc::rocshmem )
list(APPEND _cmake_import_check_files_for_roc::rocshmem "${_IMPORT_PREFIX}/lib/librocshmem.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
