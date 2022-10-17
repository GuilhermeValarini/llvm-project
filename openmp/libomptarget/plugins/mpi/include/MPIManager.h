//===---------- RTLs/mpi/src/rtl.h - MPI RTL Definition - C++ -----------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations for the MPI RTL plugin.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_OMPCLUSTER_MPI_MANAGER_H_
#define _OMPTARGET_OMPCLUSTER_MPI_MANAGER_H_

#include <list>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DynamicLibrary.h"

#include "Common.h"
#include "EventSystem.h"

#include "MemoryManager.h"
#include "omptarget.h"

using DynamicLibrary = llvm::sys::DynamicLibrary;
template <typename T> using SmallVector = llvm::SmallVector<T>;

// TODO: Maybe include that inside MPIProcessManager?
/// Array of Dynamic libraries loaded for this target.
struct DynLibTy {
  std::string FileName;
  std::unique_ptr<DynamicLibrary> DynLib;
};

// TODO: Maybe include that inside MPIProcessManager?
/// Keep entries table per device.
struct FuncOrGblEntryTy {
  __tgt_target_table Table;
  SmallVector<__tgt_offload_entry> Entries;
};

// Memory Allocator.
// ============================================================================
// TODO: Maybe include that inside MPIProcessManager?
/// A class responsible for interacting with device native runtime library to
/// allocate and free memory.
class MPIDeviceAllocatorTy : public DeviceAllocatorTy {
  const int DeviceId;
  EventSystemTy &EventSystem;

public:
  MPIDeviceAllocatorTy(int DeviceId, EventSystemTy &EventSystem)
      : DeviceId(DeviceId), EventSystem(EventSystem) {}

  void *allocate(size_t Size, void *HostPtr,
                 TargetAllocTy Kind = TARGET_ALLOC_DEFAULT) override;

  int free(void *TargetPtr, TargetAllocTy Kind = TARGET_ALLOC_DEFAULT) override;
};

// MPI Manager.
// ============================================================================
// TODO: Make not copyable nor movable. Singleton?
/// Class containing all the device information.
class MPIManagerTy {
  // Distributed event system responsible for hiding communications between
  // nodes.
  // TODO: Put this on event system file?
  using EventQueue = SmallVector<EventPtr>;
  EventSystemTy EventSystem;

  // Memory manager
  // ===========================================================================
  // Whether use memory manager
  bool UseMemoryManager = true;
  // A vector of device allocators
  SmallVector<MPIDeviceAllocatorTy> ProcessAllocators{};
  // A vector of memory managers. Since the memory manager is non-copyable and
  // non-movable, we wrap them into std::unique_ptr.
  SmallVector<std::unique_ptr<MemoryManagerTy>> MemoryManagers{};

  // Dynamic libraries and and function list
  // ===========================================================================
  std::list<DynLibTy> DynLibs{};
  SmallVector<std::list<FuncOrGblEntryTy>> FuncGblEntries{};

  // TODO: Use atomic and query for it when setting.
  bool IsInitialized = false;

  // De/initialization functions.
  // ===========================================================================
public:
  MPIManagerTy(){};
  ~MPIManagerTy();

  bool initialize();
  bool deinitialize();

  // Dynamic library loading and handling.
  // ===========================================================================
public:
  // Load a binary into a device context.
  __tgt_target_table *loadBinary(const int DeviceId,
                                 const __tgt_device_image *Image);

  // Check whether is given binary valid for the plugin.
  int32_t isValidBinary(__tgt_device_image *Image) const;

private:
  // Record entry point associated with a device.
  void createOffloadTable(int32_t DeviceId,
                          SmallVector<__tgt_offload_entry> &&Entries);

  // Return true if the entry is associated with the device.
  bool findOffloadEntry(int32_t DeviceId, void *Addr);

  // Return the pointer to the target entries table.
  __tgt_target_table *getOffloadEntriesTable(int32_t DeviceId);

  // Return the pointer to the target entries table.
  __tgt_target_table *getOffloadEntriesTableOnWorker();

  // Register the shared library to the current device.
  void registerLibOnWorker(__tgt_bin_desc *Desc);

  __tgt_target_table *loadBinaryOnWorker(const __tgt_device_image *Image);

  // Plugin and device information.
  // ===========================================================================
public:
  bool isValidDeviceId(const int DeviceId) const;

  int getNumOfDevices() const;

  // Valid check helpers.
  // ===========================================================================
private:
  bool checkValidDeviceId(const int DeviceId) const;

  bool checkValidAsyncInfo(const __tgt_async_info *AsyncInfo) const;

  int32_t checkCreatedEvent(const EventPtr &Event) const;

  bool checkRecordedEventPtr(const void *Event) const;

  // Data management.
  // ===========================================================================
public:
  void *dataAlloc(int32_t DeviceId, int64_t Size, void *HostPtr,
                  TargetAllocTy Kind);

  int32_t dataDelete(int32_t DeviceId, void *TargetPtr, TargetAllocTy Kind);

  int32_t dataSubmit(int32_t DeviceId, void *TargetPtr, void *HostPtr,
                     int64_t Size, __tgt_async_info *AsyncInfo);

  int32_t dataRetrieve(int32_t DeviceId, void *HostPtr, void *TargetPtr,
                       int64_t Size, __tgt_async_info *AsyncInfo);

  int32_t dataExchange(int32_t SrcID, void *SrcPtr, int32_t DstId, void *DstPtr,
                       int64_t Size, __tgt_async_info *AsyncInfo);

  // Target execution.
  // ===========================================================================
public:
  int32_t runTargetRegion(int32_t DeviceId, void *Entry, void **Args,
                          ptrdiff_t *Offsets, int32_t NumArgs,
                          __tgt_async_info *AsyncInfo);

  // Event queueing and synchronization.
  // ===========================================================================
public:
  int32_t synchronize(int32_t DeviceId, __tgt_async_info *AsyncInfo);

private:
  // Acquire the async context from the async info object. If no context is
  // present, a new one is created.
  EventQueue *getEventQueue(__tgt_async_info *AsyncInfo);

  // Push a new event to the respective device queue, updating the async info
  // context.
  void pushNewEvent(const EventPtr &Event, __tgt_async_info *AsyncInfo);

  // Device side functions.
  // ===========================================================================
public:
  // Return true if currently being executed inside the device.
  bool isInsideDevice();

  // Start device main for worker ranks.
  void runDeviceMain(__tgt_bin_desc *Desc);

  // External events management.
  // ===========================================================================
public:
  // Allocates a shared pointer to an event.
  int32_t createEvent(int32_t DeviceId, void **Event);

  // Destroys a shared pointer to an event.
  int32_t destroyEvent(int32_t DeviceId, void *Event);

  // Binds Event to the last internal event present in the event queue.
  int32_t recordEvent(int32_t DeviceId, void *Event,
                      __tgt_async_info *AsyncInfo);

  // Adds the `Event` to the event queue so we can wait for it. `Event` might
  // come from another device event queue (even on another task context),
  // allowing two tasks to synchronize their inner events when needed (e.g.:
  // wait for a data to be submitted).
  int32_t waitEvent(int32_t DeviceId, void *Event, __tgt_async_info *AsyncInfo);

  // Waits for the Event, blocking the caller thread.
  int32_t syncEvent(int32_t DeviceId, void *Event);
};

#endif // _OMPTARGET_OMPCLUSTER_MPI_MANAGER_H_
