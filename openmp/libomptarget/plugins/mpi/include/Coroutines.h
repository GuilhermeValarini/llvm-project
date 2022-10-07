//===------- coroutines.h - Concurrent MPI communicaiton ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions to implement coroutines in C++17.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_OMPCLUSTER_COROUTINES_H_
#define _OMPTARGET_OMPCLUSTER_COROUTINES_H_

// Support macros
#define CONCAT_IMPL(x, y) x##y
#define MACRO_CONCAT(x, y) CONCAT_IMPL(x, y)
#define GET_MACRO(_1, name, ...) name

// Coroutine control macros
// NOTE: Please, leave the macros definition in the same line or split the
// macros using `\`.
// =============================================================================
// Begin a coroutine function.
#define CO_BEGIN()                                                             \
  do {                                                                         \
    if (ResumeLocation != nullptr) {                                           \
      goto *ResumeLocation;                                                    \
    }                                                                          \
  } while (false);
// End a coroutine function, maybe returning a final value.
#define CO_RETURN_VOID()                                                       \
  do {                                                                         \
    ResumeLocation = &&MACRO_CONCAT(COROUTINE_YIELD_, __LINE__);               \
    MACRO_CONCAT(COROUTINE_YIELD_, __LINE__) :;                                \
    return;                                                                    \
  } while (false);
#define CO_RETURN_VALUE(value)                                                 \
  do {                                                                         \
    ResumeLocation = &&MACRO_CONCAT(COROUTINE_YIELD_, __LINE__);               \
    MACRO_CONCAT(COROUTINE_YIELD_, __LINE__) :;                                \
    return value;                                                              \
  } while (false);
#define CO_RETURN(...)                                                         \
  GET_MACRO(__VA_ARGS__, CO_RETURN_VALUE, CO_RETURN_VOID)(__VA_ARGS__)
// Halts the coroutine execution. The next call will resume the execution
#define CO_YIELD_VOID()                                                        \
  do {                                                                         \
    ResumeLocation = &&MACRO_CONCAT(COROUTINE_YIELD_, __LINE__);               \
    return;                                                                    \
    MACRO_CONCAT(COROUTINE_YIELD_, __LINE__) :;                                \
  } while (false);
#define CO_YIELD_VALUE(value)                                                  \
  do {                                                                         \
    ResumeLocation = &&MACRO_CONCAT(COROUTINE_YIELD_, __LINE__);               \
    return value;                                                              \
    MACRO_CONCAT(COROUTINE_YIELD_, __LINE__) :;                                \
  } while (false);
#define CO_YIELD(...)                                                          \
  GET_MACRO(__VA_ARGS__, CO_YIELD_VALUE, CO_YIELD_VOID)(__VA_ARGS__)

// TODO: Refactor the event system to use this interface.
// // Coroutine base structure.
// template <typename ReturnType> struct Coroutine {
//   using LabelPointer = void *;
//   LabelPointer ResumeLocation = nullptr;

//   virtual ReturnType operator()() = 0;
// };

#endif // _OMPTARGET_OMPCLUSTER_COROUTINES_H_
