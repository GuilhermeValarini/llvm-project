##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build a plugin for a MPI machine if available.
#
##===----------------------------------------------------------------------===##
if (NOT(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux"))
  libomptarget_say("Not building MPI offloading plugin: only support MPI in Linux x86_64 or ppc64le hosts.")
  return()
elseif (NOT LIBOMPTARGET_DEP_LIBFFI_FOUND)
  libomptarget_say("Not building MPI offloading plugin: libffi dependency not found.")
  return()
elseif(NOT LIBOMPTARGET_DEP_MPI_FOUND)
  libomptarget_say("Not building MPI offloading plugin: MPI not found in system.")
  return()
endif()

libomptarget_say("Building MPI offloading plugin.")

# Source defines.
# ==============================================================================
# Define the suffix for the runtime messaging dumps.
add_definitions(-DTARGET_NAME=MPI)
# Define macro with the ELF ID for this target.
add_definitions("-DTARGET_ELF_ID=62")

add_llvm_library(omptarget.rtl.mpi SHARED
  src/EventSystem.cpp
  src/MPIManager.cpp
  src/rtl.cpp

  ADDITIONAL_HEADER_DIRS
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_DEP_MPI_INCLUDE_DIRS}
  ${LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIRS}

  LINK_LIBS
  PRIVATE
  ${LIBOMPTARGET_DEP_MPI_LIBRARIES}
  ${LIBOMPTARGET_DEP_LIBFFI_LIBRARIES}
  elf_common
  MemoryManager
  dl
  ${OPENMP_PTHREAD_LIB}
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports"
  ${LIBOMPTARGET_DEP_MPI_LINK_FLAGS}

  NO_INSTALL_RPATH
)

# Add include directories
target_include_directories(omptarget.rtl.mpi
  PRIVATE include ${LIBOMPTARGET_INCLUDE_DIR}
)

# Install plugin under the lib destination folder.
install(TARGETS omptarget.rtl.mpi LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")
set_target_properties(omptarget.rtl.mpi PROPERTIES 
  INSTALL_RPATH "$ORIGIN" BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")

if(LIBOMPTARGET_DEP_MPI_COMPILE_FLAGS)
  set_target_properties(omptarget.rtl.mpi PROPERTIES
                        COMPILE_FLAGS "${LIBOMPTARGET_DEP_MPI_COMPILE_FLAGS}")
endif()

# Report to the parent scope that we are building a plugin for MPI.
set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS} x86_64-pc-linux-gnu-ompc" PARENT_SCOPE)
