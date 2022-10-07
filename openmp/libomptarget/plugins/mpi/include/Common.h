//===------- coroutines.h - Concurrent MPI communicaiton ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains common functionalities for MPI plugin
//
//===----------------------------------------------------------------------===//

#include <cassert>

// Debug utilities definitions
// ===========================================================================
#ifndef TARGET_NAME
#define TARGET_NAME MPI
#endif
#define DEBUG_PREFIX "Target " GETNAME(TARGET_NAME) " RTL"
#include "Debug.h"

#define CHECK(expr, msg, ...)                                                  \
  if (!(expr)) {                                                               \
    REPORT(msg, ##__VA_ARGS__);                                                \
    return false;                                                              \
  }

#define assertm(expr, msg) assert(((void)msg, expr));
