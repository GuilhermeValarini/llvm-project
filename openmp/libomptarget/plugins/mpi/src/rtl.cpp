//===------RTLs/mpi/src/rtl.cpp - Target RTLs Implementation - C++ ------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL for MPI applications
//
//===----------------------------------------------------------------------===//

#include <functional>
#include <type_traits>
#include <utility>

#include "Common.h"
#include "MPIManager.h"

#include "omptargetplugin.h"

/// Global context that stores device information for the entire binary.
static MPIManagerTy *MPIManager = nullptr;

/// Helper functions
template <typename FuncTy, typename... ArgsTys>
static int32_t syncCall(FuncTy Function, int32_t DeviceId, ArgsTys &&...Args) {
  static_assert(std::is_member_function_pointer_v<FuncTy>,
                "FuncTy should be a member function pointer.");
  using IDTy = int32_t;
  using AsyncInfoPtrTy = __tgt_async_info *;
  static_assert(std::is_invocable_r_v<int32_t, FuncTy, MPIManagerTy, IDTy,
                                      ArgsTys..., AsyncInfoPtrTy>,
                "Function is not callable with given arguments.");

  __tgt_async_info LocalAsyncInfo;
  int32_t RetCode =
      std::invoke(Function, MPIManager, DeviceId,
                  std::forward<ArgsTys>(Args)..., &LocalAsyncInfo);

  if (RetCode == OFFLOAD_SUCCESS)
    RetCode = MPIManager->synchronize(DeviceId, &LocalAsyncInfo);

  return RetCode;
}

#ifdef __cplusplus
extern "C" {
#endif

// De/initialization functions.
// =============================================================================

int32_t __tgt_rtl_init_plugin() {
  MPIManager = new MPIManagerTy();

  if (!MPIManager) {
    REPORT("Failed to allocate MPI plugin manager.\n");
    return OFFLOAD_FAIL;
  }

  if (!MPIManager->initialize()) {
    REPORT("Failed to initialize the MPI plugin.\n");
    delete MPIManager;
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_deinit_plugin() {
  if (!MPIManager) {
    DP("MPI plugin is not initialized. Deinitialization will do nothing.\n");
    return OFFLOAD_FAIL;
  }

  int32_t Ret = OFFLOAD_SUCCESS;

  if (!MPIManager->deinitialize()) {
    REPORT("Failed to deinitialize the MPI plugin.\n");
    Ret = OFFLOAD_FAIL;
  }

  delete MPIManager;
  MPIManager = nullptr;

  return Ret;
}

int32_t __tgt_rtl_init_device(int32_t DeviceId) { return OFFLOAD_SUCCESS; }

int32_t __tgt_rtl_deinit_device(int32_t DeviceId) { return OFFLOAD_SUCCESS; }

// Dynamic library loading and handling.
// =============================================================================

int32_t __tgt_rtl_is_valid_binary(__tgt_device_image *Image) {
  return MPIManager->isValidBinary(Image);
}

__tgt_target_table *__tgt_rtl_load_binary(int32_t DeviceId,
                                          __tgt_device_image *Image) {
  return MPIManager->loadBinary(DeviceId, Image);
}

// Plugin and device information.
// =============================================================================

int32_t __tgt_rtl_number_of_devices() { return MPIManager->getNumOfDevices(); }

int32_t __tgt_rtl_is_data_exchangable(int32_t SrcDevId, int32_t DstDevId) {
  return MPIManager->isValidDeviceId(SrcDevId) &&
         MPIManager->isValidDeviceId(DstDevId);
}

// Plugin and device configuration.
// =============================================================================

// TODO: check what this could be used for.
// TODO: Used to control log level
void __tgt_rtl_set_info_flag(uint32_t) { return; }

// Data management.
// ===========================================================================

void *__tgt_rtl_data_alloc(int32_t DeviceId, int64_t Size, void *HostPtr,
                           int32_t Kind) {
  return MPIManager->dataAlloc(DeviceId, Size, HostPtr, (TargetAllocTy)Kind);
}

int32_t __tgt_rtl_data_delete(int32_t DeviceId, void *TargetPtr, int32_t Kind) {
  return MPIManager->dataDelete(DeviceId, TargetPtr, (TargetAllocTy)Kind);
}

int32_t __tgt_rtl_data_submit(int32_t DeviceId, void *TargetPtr, void *HostPtr,
                              int64_t Size) {
  return syncCall(&MPIManagerTy::dataSubmit, DeviceId, TargetPtr, HostPtr,
                  Size);
}

int32_t __tgt_rtl_data_submit_async(int32_t DeviceId, void *TargetPtr,
                                    void *HostPtr, int64_t Size,
                                    __tgt_async_info *AsyncInfo) {
  return MPIManager->dataSubmit(DeviceId, TargetPtr, HostPtr, Size, AsyncInfo);
}

int32_t __tgt_rtl_data_retrieve(int32_t DeviceId, void *HostPtr,
                                void *TargetPtr, int64_t Size) {
  return syncCall(&MPIManagerTy::dataRetrieve, DeviceId, HostPtr, TargetPtr,
                  Size);
}

int32_t __tgt_rtl_data_retrieve_async(int32_t DeviceId, void *HostPtr,
                                      void *TargetPtr, int64_t Size,
                                      __tgt_async_info *AsyncInfo) {
  return MPIManager->dataRetrieve(DeviceId, HostPtr, TargetPtr, Size,
                                  AsyncInfo);
}

int32_t __tgt_rtl_data_exchange(int32_t SrcId, void *SrcPtr, int32_t DstId,
                                void *DstPtr, int64_t Size) {
  return syncCall(&MPIManagerTy::dataExchange, SrcId, SrcPtr, DstId, DstPtr,
                  Size);
}

int32_t __tgt_rtl_data_exchange_async(int32_t SrcId, void *SrcPtr,
                                      int32_t DstId, void *DstPtr, int64_t Size,
                                      __tgt_async_info *AsyncInfo) {
  return MPIManager->dataExchange(SrcId, SrcPtr, DstId, DstPtr, Size,
                                  AsyncInfo);
}

// Target execution.
// ===========================================================================

int32_t __tgt_rtl_run_target_region(int32_t DeviceId, void *Entry, void **Args,
                                    ptrdiff_t *Offsets, int32_t NumArgs) {
  return syncCall(&MPIManagerTy::runTargetRegion, DeviceId, Entry, Args,
                  Offsets, NumArgs);
}

int32_t __tgt_rtl_run_target_region_async(int32_t DeviceId, void *Entry,
                                          void **Args, ptrdiff_t *Offsets,
                                          int32_t NumArgs,
                                          __tgt_async_info *AsyncInfo) {
  return MPIManager->runTargetRegion(DeviceId, Entry, Args, Offsets, NumArgs,
                                     AsyncInfo);
}

int32_t __tgt_rtl_run_target_team_region(int32_t DeviceId, void *Entry,
                                         void **Args, ptrdiff_t *Offsets,
                                         int32_t NumArgs, int32_t NumTeams,
                                         int32_t ThreadLimit,
                                         uint64_t LoopTripcount) {
  return __tgt_rtl_run_target_region(DeviceId, Entry, Args, Offsets, NumArgs);
}

int32_t __tgt_rtl_run_target_team_region_async(
    int32_t DeviceId, void *Entry, void **Args, ptrdiff_t *Offsets,
    int32_t NumArgs, int32_t NumTeams, int32_t ThreadLimit,
    uint64_t LoopTripCount, __tgt_async_info *AsyncInfo) {
  return __tgt_rtl_run_target_region_async(DeviceId, Entry, Args, Offsets,
                                           NumArgs, AsyncInfo);
}

// Asynchronous context management.
// ===========================================================================

int32_t __tgt_rtl_synchronize(int32_t DeviceId, __tgt_async_info *AsyncInfo) {
  return MPIManager->synchronize(DeviceId, AsyncInfo);
}

// External events management.
// ===========================================================================

int32_t __tgt_rtl_create_event(int32_t DeviceId, void **Event) {
  return MPIManager->createEvent(DeviceId, Event);
}

int32_t __tgt_rtl_record_event(int32_t DeviceId, void *Event,
                               __tgt_async_info *AsyncInfo) {
  return MPIManager->recordEvent(DeviceId, Event, AsyncInfo);
}

int32_t __tgt_rtl_wait_event(int32_t DeviceId, void *Event,
                             __tgt_async_info *AsyncInfo) {
  return MPIManager->waitEvent(DeviceId, Event, AsyncInfo);
}

int32_t __tgt_rtl_sync_event(int32_t DeviceId, void *Event) {
  return MPIManager->syncEvent(DeviceId, Event);
}

int32_t __tgt_rtl_destroy_event(int32_t DeviceId, void *Event) {
  return MPIManager->destroyEvent(DeviceId, Event);
}

// Device side operations.
// ===========================================================================

int32_t __tgt_rtl_is_inside_device() { return MPIManager->isInsideDevice(); }

void __tgt_rtl_run_device_main(__tgt_bin_desc *Desc) {
  MPIManager->runDeviceMain(Desc);
}

#ifdef __cplusplus
}
#endif
