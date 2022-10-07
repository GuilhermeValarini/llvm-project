//===------RTLs/mpi/src/rtl.cpp - Target RTLs Implementation - C++ ------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL for MPI machine
//
//===----------------------------------------------------------------------===//

#include "MPIManager.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <ffi.h>
#include <gelf.h>
#include <link.h>
#include <string>
#include <vector>

// ELF utilities definitions
// ===========================================================================
#ifndef TARGET_ELF_ID
#define TARGET_ELF_ID 0
#endif
#include "elf_common.h"

// Memory Allocator
// ===========================================================================
constexpr const char *toString(const TargetAllocTy &Kind) {
  switch (Kind) {
  case TargetAllocTy::TARGET_ALLOC_DEFAULT:
    return "TARGET_ALLOC_DEFAULT";
  case TargetAllocTy::TARGET_ALLOC_DEVICE:
    return "TARGET_ALLOC_DEVICE";
  case TargetAllocTy::TARGET_ALLOC_HOST:
    return "TARGET_ALLOC_HOST";
  case TargetAllocTy::TARGET_ALLOC_SHARED:
    return "TARGET_ALLOC_SHARED";
  }
}

void *MPIDeviceAllocatorTy::allocate(size_t Size, void *HostPtr,
                                     TargetAllocTy Kind) {
  if (Kind != TargetAllocTy::TARGET_ALLOC_DEFAULT &&
      Kind != TargetAllocTy::TARGET_ALLOC_DEVICE) {
    REPORT("Invalid allocation kind %s. MPI plugin only supports "
           "TARGET_ALLOC_DEFAULT and TARGET_ALLOC_DEVICE.",
           toString(Kind));
    return nullptr;
  }

  if (Size == 0)
    return nullptr;

  void *DevicePtr = nullptr;
  auto event =
      EventSystem.createEvent<AllocEventTy>(DeviceId, Size, &DevicePtr);
  event->wait();

  if (event->getEventState() == EventStateTy::FAILED)
    return nullptr;

  return DevicePtr;
}

int MPIDeviceAllocatorTy::free(void *TargetPtr, TargetAllocTy Kind) {
  if (Kind != TargetAllocTy::TARGET_ALLOC_DEFAULT &&
      Kind != TargetAllocTy::TARGET_ALLOC_DEVICE) {
    REPORT("Invalid allocation kind %s. MPI plugin only supports "
           "TARGET_ALLOC_DEFAULT and TARGET_ALLOC_DEVICE.",
           toString(Kind));
    return OFFLOAD_FAIL;
  }

  auto event = EventSystem.createEvent<DeleteEventTy>(DeviceId, TargetPtr);
  event->wait();

  if (event->getEventState() == EventStateTy::FAILED)
    return OFFLOAD_FAIL;

  return OFFLOAD_SUCCESS;
}

// MPI Manager
// ===========================================================================

MPIManagerTy::MPIManagerTy() : EventSystem() {
  const int NumWorkers = EventSystem.getNumWorkers();

  // Set function entries
  FuncGblEntries.resize(EventSystem.isHead() ? EventSystem.getNumWorkers() : 1);

  // Set device allocators and the memory manager.
  for (int DeviceId = 0; DeviceId < NumWorkers; ++DeviceId)
    ProcessAllocators.emplace_back(DeviceId, EventSystem);

  auto [ManagerThreshold, ManagerEnabled] =
      MemoryManagerTy::getSizeThresholdFromEnv();
  UseMemoryManager = ManagerEnabled;

  if (UseMemoryManager) {
    for (int i = 0; i < NumWorkers; ++i)
      MemoryManagers.emplace_back(std::make_unique<MemoryManagerTy>(
          ProcessAllocators[i], ManagerThreshold));
  }
}

MPIManagerTy::~MPIManagerTy() {
  // Close dynamic libraries
  for (auto &Lib : DynLibs) {
    if (Lib.DynLib->isValid())
      remove(Lib.FileName.c_str());
  }

  // Destruct memory managers
  for (auto &m : MemoryManagers)
    m.release();
}

void MPIManagerTy::createOffloadTable(
    int32_t DeviceId, SmallVector<__tgt_offload_entry> &&Entries) {
  assert(DeviceId < (int32_t)FuncGblEntries.size() && "Unexpected device id!");
  FuncGblEntries[DeviceId].emplace_back();
  FuncOrGblEntryTy &E = FuncGblEntries[DeviceId].back();

  E.Entries = Entries;
  E.Table.EntriesBegin = E.Entries.begin();
  E.Table.EntriesEnd = E.Entries.end();
}

bool MPIManagerTy::findOffloadEntry(int32_t DeviceId, void *Addr) {
  assert(DeviceId < (int32_t)FuncGblEntries.size() && "Unexpected device id!");
  FuncOrGblEntryTy &E = FuncGblEntries[DeviceId].back();

  for (__tgt_offload_entry *I = E.Table.EntriesBegin, *End = E.Table.EntriesEnd;
       I < End; ++I) {
    if (I->addr == Addr)
      return true;
  }

  return false;
}

__tgt_target_table *MPIManagerTy::getOffloadEntriesTable(int32_t DeviceId) {
  assert(DeviceId < (int32_t)FuncGblEntries.size() && "Unexpected device id!");
  FuncOrGblEntryTy &E = FuncGblEntries[DeviceId].back();

  return &E.Table;
}

__tgt_target_table *MPIManagerTy::getOffloadEntriesTableOnWorker() {
  return getOffloadEntriesTable(0);
}

void MPIManagerTy::registerLib(__tgt_bin_desc *Desc) {
  // Register the images with the RTLs that understand them, if any.
  for (int32_t I = 0; I < Desc->NumDeviceImages; ++I) {
    __tgt_device_image *Img = &Desc->DeviceImages[I];

    if (!isValidBinary(Img)) {
      REPORT("Image " DPxMOD " is NOT compatible with this MPI device!\n",
             DPxPTR(Img->ImageStart));
      continue;
    }

    DP("Image " DPxMOD " is compatible with this MPI device!\n",
       DPxPTR(Img->ImageStart));

    loadBinaryOnWorker(Img);
  }

  DP("Done registering entries!\n");
}

int32_t MPIManagerTy::isValidBinary(__tgt_device_image *Image) const {
// If we don't have a valid ELF ID we can just fail.
#if TARGET_ELF_ID < 1
  return 0;
#else
  return elf_check_machine(Image, TARGET_ELF_ID);
#endif
}

__tgt_target_table *MPIManagerTy::loadBinary(const int DeviceId,
                                             const __tgt_device_image *Image) {
  DP("Dev %d: load binary from " DPxMOD " image\n", DeviceId,
     DPxPTR(Image->ImageStart));

  if (checkValidDeviceId(DeviceId))
    return nullptr;

  size_t ImageSize = (size_t)Image->ImageEnd - (size_t)Image->ImageStart;

  // load dynamic library and get the entry points. We use the dl library
  // to do the loading of the library, but we could do it directly to avoid the
  // dump to the temporary file.
  //
  // 1) Create tmp file with the library contents.
  // 2) Use dlopen to load the file and dlsym to retrieve the symbols.
  char TmpName[] = "/tmp/tmpfile_XXXXXX";
  int TmpFd = mkstemp(TmpName);

  if (TmpFd == -1) {
    REPORT("Failed to load binary. Failed to create temporary file %s.\n",
           TmpName);
    return nullptr;
  }

  FILE *Ftmp = fdopen(TmpFd, "wb");

  if (!Ftmp) {
    REPORT("Failed to load binary. Failed to open new temporary file %s.\n",
           TmpName);
    return nullptr;
  }

  fwrite(Image->ImageStart, ImageSize, 1, Ftmp);
  fclose(Ftmp);

  std::string ErrMsg;
  auto DynLib = std::make_unique<DynamicLibrary>(
      llvm::sys::DynamicLibrary::getPermanentLibrary(TmpName, &ErrMsg));
  DynLibTy Lib = {TmpName, std::move(DynLib)};

  if (!Lib.DynLib->isValid()) {
    REPORT("Failed to load binary. Loading error: %s.\n", ErrMsg.c_str());
    return nullptr;
  }

  __tgt_offload_entry *HostBegin = Image->EntriesBegin;
  __tgt_offload_entry *HostEnd = Image->EntriesEnd;

  // Create a new offloading entry list using the device symbol address.
  SmallVector<__tgt_offload_entry> Entries;
  for (__tgt_offload_entry *E = HostBegin; E != HostEnd; ++E) {
    if (!E->addr) {
      REPORT("Failed to load binary. Found null entry.\n");
      return nullptr;
    }

    __tgt_offload_entry Entry = *E;

    void *DevAddr = Lib.DynLib->getAddressOfSymbol(E->name);
    Entry.addr = DevAddr;

    DP("Entry point " DPxMOD " maps to global %s (" DPxMOD ")\n",
       DPxPTR(E - HostBegin), E->name, DPxPTR(DevAddr));

    Entries.emplace_back(Entry);
  }

  createOffloadTable(DeviceId, std::move(Entries));
  DynLibs.emplace_back(std::move(Lib));

  return getOffloadEntriesTable(DeviceId);
}

__tgt_target_table *
MPIManagerTy::loadBinaryOnWorker(const __tgt_device_image *Image) {
  return loadBinary(0, Image);
}

bool MPIManagerTy::isValidDeviceId(const int DeviceId) const {
  return DeviceId >= 0 && DeviceId < EventSystem.getNumWorkers();
}

int MPIManagerTy::getNumOfDevices() const {
  return EventSystem.getNumWorkers();
}

bool MPIManagerTy::checkValidDeviceId(const int DeviceId) const {
  if (!isValidDeviceId(DeviceId)) {
    REPORT("Failed to load binary. Received device id %d out of range of valid "
           "ids [%d, %d]\n",
           DeviceId, 0, EventSystem.getNumWorkers());
    return false;
  }

  return true;
}

bool MPIManagerTy::checkValidAsyncInfo(
    const __tgt_async_info *AsyncInfo) const {
  if (AsyncInfo == nullptr) {
    REPORT("Plugin call failed. Received null AsyncInfo\n");
    return false;
  }

  return true;
}

int32_t MPIManagerTy::checkCreatedEvent(const EventPtr &Event) const {
  if (Event->getEventState() != EventStateTy::CREATED)
    return OFFLOAD_FAIL;

  return OFFLOAD_SUCCESS;
}

bool MPIManagerTy::checkRecordedEventPtr(const void *Event) const {
  if (!Event) {
    REPORT("Received an invalid recorded event pointer\n");
    return false;
  }

  return true;
}

void *MPIManagerTy::dataAlloc(int32_t DeviceId, int64_t Size, void *HostPtr,
                              TargetAllocTy Kind) {
  if (!checkValidDeviceId(DeviceId))
    return nullptr;

  if (UseMemoryManager)
    return MemoryManagers[DeviceId]->allocate(Size, HostPtr);

  return ProcessAllocators[DeviceId].allocate(Size, HostPtr, Kind);
}

int32_t MPIManagerTy::dataDelete(int32_t DeviceId, void *TargetPtr,
                                 TargetAllocTy Kind) {
  if (!checkValidDeviceId(DeviceId))
    return OFFLOAD_FAIL;

  if (UseMemoryManager)
    return MemoryManagers[DeviceId]->free(TargetPtr);

  return ProcessAllocators[DeviceId].free(TargetPtr, Kind);
}

int32_t MPIManagerTy::dataSubmit(int32_t DeviceId, void *TargetPtr,
                                 void *HostPtr, int64_t Size,
                                 __tgt_async_info *AsyncInfo) {
  if (!checkValidDeviceId(DeviceId))
    return OFFLOAD_FAIL;

  if (!checkValidAsyncInfo(AsyncInfo))
    return OFFLOAD_FAIL;

  auto Event = EventSystem.createEvent<SubmitEventTy>(DeviceId, TargetPtr,
                                                      HostPtr, Size);
  pushNewEvent(Event, AsyncInfo);

  return checkCreatedEvent(Event);
}

int32_t MPIManagerTy::dataRetrieve(int32_t DeviceId, void *HostPtr,
                                   void *TargetPtr, int64_t Size,
                                   __tgt_async_info *AsyncInfo) {
  if (!checkValidDeviceId(DeviceId))
    return OFFLOAD_FAIL;

  if (!checkValidAsyncInfo(AsyncInfo))
    return OFFLOAD_FAIL;

  auto Event = EventSystem.createEvent<RetrieveEventTy>(DeviceId, HostPtr,
                                                        TargetPtr, Size);
  pushNewEvent(Event, AsyncInfo);

  return checkCreatedEvent(Event);
}

int32_t MPIManagerTy::dataExchange(int32_t SrcID, void *SrcPtr, int32_t DstId,
                                   void *DstPtr, int64_t Size,
                                   __tgt_async_info *AsyncInfo) {
  if (!checkValidDeviceId(SrcID) || !checkValidDeviceId(DstId))
    return OFFLOAD_FAIL;

  if (!checkValidAsyncInfo(AsyncInfo))
    return OFFLOAD_FAIL;

  auto Event = EventSystem.createEvent<ExchangeEventTy>(SrcID, DstId + 1,
                                                        SrcPtr, DstPtr, Size);
  pushNewEvent(Event, AsyncInfo);

  return OFFLOAD_SUCCESS;
}

int32_t MPIManagerTy::runTargetRegion(int32_t DeviceId, void *Entry,
                                      void **Args, ptrdiff_t *Offsets,
                                      int32_t NumArgs,
                                      __tgt_async_info *AsyncInfo) {
  if (!checkValidDeviceId(DeviceId))
    return OFFLOAD_FAIL;

  if (!checkValidAsyncInfo(AsyncInfo))
    return OFFLOAD_FAIL;

  // Prepare all args based on their offsets.
  SmallVector<void *> ArgPtrs(NumArgs);

  for (int I = 0; I < NumArgs; ++I) {
    ArgPtrs[I] = (void *)((intptr_t)Args[I] + Offsets[I]);
  }

  // get the translation table (which contains all the good info).
  __tgt_target_table *TargetTable = getOffloadEntriesTable(DeviceId);
  // iterate over all the host table entries to see if we can locate the
  // host_ptr.
  __tgt_offload_entry *Begin = TargetTable->EntriesBegin;
  __tgt_offload_entry *End = TargetTable->EntriesEnd;
  __tgt_offload_entry *Curr = Begin;

  uint32_t EntryIdx = -1;

  for (uint32_t I = 0; Curr < End; ++Curr, ++I) {
    if (Curr->addr != Entry)
      continue;
    // we got a match, now fill the HostPtrToTableMap so that we
    // may avoid this search next time.
    DP("[MPI host] Running kernel called %s...\n", Curr->name);
    EntryIdx = I;
    break;
  }

  auto event = EventSystem.createEvent<ExecuteEventTy>(
      DeviceId, NumArgs, ArgPtrs.data(), EntryIdx);
  pushNewEvent(event, AsyncInfo);

  return OFFLOAD_SUCCESS;
}

MPIManagerTy::EventQueue *
MPIManagerTy::getEventQueue(__tgt_async_info *AsyncInfo) {
  if (!checkValidAsyncInfo(AsyncInfo))
    return nullptr;

  auto Queue = new EventQueue;
  if (AsyncInfo->Queue == nullptr) {
    AsyncInfo->Queue = reinterpret_cast<void *>(Queue);
  }

  return Queue;
}

void MPIManagerTy::pushNewEvent(const EventPtr &Event,
                                __tgt_async_info *AsyncInfo) {
  auto *Queue = getEventQueue(AsyncInfo);
  Queue->push_back(Event);
}

int32_t MPIManagerTy::synchronize(int32_t DeviceId,
                                  __tgt_async_info *AsyncInfo) {
  if (AsyncInfo == nullptr || AsyncInfo->Queue == nullptr)
    return OFFLOAD_SUCCESS;

  // Acquire the async context.
  EventQueue *Queue = getEventQueue(AsyncInfo);

  int Result = OFFLOAD_SUCCESS;
  for (auto &Event : *Queue) {
    Event->wait();

    // Check if the event failed
    if (Event->getEventState() == EventStateTy::FAILED) {
      REPORT("Event %s has failed during synchronization.\n",
             toString(Event->EventType));
      Result = OFFLOAD_FAIL;
      break;
    }
  }

  // Delete the current async_info context. Further use of the same async_info
  // object must create a new context.
  delete Queue;
  AsyncInfo->Queue = nullptr;

  return Result;
}

////////////////////////////////////////////////////////////////////////////////
/// Start device main for worker ranks
int32_t MPIManagerTy::startDeviceMain(__tgt_bin_desc *desc) {
  // Check whether it is a device or not and if so run its initialization
  if (EventSystem.isHead())
    return OFFLOAD_SUCCESS;

  DP("Running main function on workers\n");

  registerLib(desc);

  EventSystem.runGateThread(getOffloadEntriesTableOnWorker());

  DP("Exiting main function on workers\n");

  // Will call the destructor of all static objects, thus all the MPI
  // finalization code will be called.
  exit(0);
}

// Synchronization event management
// ===========================================================================
int32_t MPIManagerTy::createEvent(int32_t ID, void **Event) {
  if (checkRecordedEventPtr(Event))
    return OFFLOAD_FAIL;

  auto RecordedEvent = new EventPtr;
  if (RecordedEvent == nullptr) {
    REPORT("Could not allocate a new synchronization event\n");
    return OFFLOAD_FAIL;
  }

  *Event = reinterpret_cast<void *>(RecordedEvent);

  return OFFLOAD_SUCCESS;
}

int32_t MPIManagerTy::destroyEvent(int32_t ID, void *Event) {
  if (checkRecordedEventPtr(Event))
    return OFFLOAD_FAIL;

  delete reinterpret_cast<EventPtr *>(Event);

  return OFFLOAD_SUCCESS;
}

int32_t MPIManagerTy::recordEvent(int32_t ID, void *Event,
                                  __tgt_async_info *AsyncInfo) {
  if (checkRecordedEventPtr(Event))
    return OFFLOAD_FAIL;

  if (AsyncInfo == nullptr || AsyncInfo->Queue == nullptr) {
    REPORT("Received an invalid async queue on recordEvent\n");
    return OFFLOAD_FAIL;
  }

  EventQueue *Queue = getEventQueue(AsyncInfo);
  if (Queue->empty()) {
    DP("Tried to record an event for an empty event queue\n");
    return OFFLOAD_SUCCESS;
  }

  // Copy the last event in the queue to the event handle.
  auto &RecordedEvent = *reinterpret_cast<EventPtr *>(Event);
  RecordedEvent = Queue->back();

  return OFFLOAD_SUCCESS;
}

int32_t MPIManagerTy::waitEvent(int32_t ID, void *Event,
                                __tgt_async_info *AsyncInfo) {
  if (checkRecordedEventPtr(Event))
    return OFFLOAD_FAIL;

  if (AsyncInfo == nullptr) {
    REPORT("Received an invalid async info on waitEvent\n");
    return OFFLOAD_FAIL;
  }

  auto &RecordedEvent = *reinterpret_cast<EventPtr *>(Event);
  if (!RecordedEvent) {
    DP("Tried to wait an empty event\n");
    return OFFLOAD_SUCCESS;
  }

  // Create a wait event that waits for `Event` to be completed and add it to
  // the event queue. This ensures that the whole event queue where `Event`
  // originated is completed up to the `Event` itself. Directly waiting on
  // `Event` would execute it instead of waiting for its predecessors in its
  // original event queue.
  EventQueue *Queue = getEventQueue(AsyncInfo);
  Queue->push_back(std::make_shared<SyncEventTy>(RecordedEvent));
  return OFFLOAD_SUCCESS;
}

int32_t MPIManagerTy::syncEvent(int32_t ID, void *Event) {
  if (checkRecordedEventPtr(Event))
    return OFFLOAD_FAIL;

  auto &RecordedEvent = *reinterpret_cast<EventPtr *>(Event);
  if (!RecordedEvent) {
    DP("Tried to synchronize an empty event\n");
    return OFFLOAD_SUCCESS;
  }

  // Create a wait event that waits for `Event` to be completed and executes it.
  // This ensures that the whole event queue where `Event` originated is
  // completed up to the `Event` itself. Directly waiting on `Event` would
  // execute it instead of waiting for its predecessors in its original event
  // queue.
  auto WaitEvent = std::make_shared<SyncEventTy>(RecordedEvent);
  WaitEvent->wait();

  return OFFLOAD_SUCCESS;
}
