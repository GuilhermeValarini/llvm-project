//===------ event_system.cpp - Concurrent MPI communication -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the MPI Event System used by the MPI
// target runtime for concurrent communication.
//
//===----------------------------------------------------------------------===//

#include "EventSystem.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ffi.h>
#include <limits>
#include <sstream>

#include "Common.h"
#include "Coroutines.h"

#include "omptarget.h"

// Helper coroutine macros
#define EVENT_BEGIN() CO_BEGIN()
#define EVENT_PAUSE() CO_YIELD(false)
#define EVENT_PAUSE_FOR_REQUESTS()                                             \
  while (!checkPendingRequests()) {                                            \
    EVENT_PAUSE();                                                             \
  }
#define EVENT_END() CO_RETURN(true)

// Customizable parameters of the event system
// =============================================================================
// Here we declare some configuration variables with their respective default
// values. Every single one of them can be tuned by an environment variable with
// the following name pattern: OMPCLUSTER_VAR_NAME.
namespace config {
// Maximum buffer Size to use during data transfer.
static int64_t MPI_FRAGMENT_SIZE = 100e6;
// Number of execute event handlers to spawn.
static int NUM_EXEC_EVENT_HANDLERS = 1;
// Number of data event handlers to spawn.
static int NUM_DATA_EVENT_HANDLERS = 1;
// Polling rate period (us) used by event handlers.
static int EVENT_POLLING_RATE = 1;
// Number of communicators to be spawned and distributed for the events. Allows
// for parallel use of network resources.
static int64_t NUM_EVENT_COMM = 10;
} // namespace config

// Helper functions
// =============================================================================
const char *toString(EventTypeTy Type) {
  switch (Type) {
  case EventTypeTy::ALLOC:
    return "Alloc";
  case EventTypeTy::DELETE:
    return "Delete";
  case EventTypeTy::RETRIEVE:
    return "Retrieve";
  case EventTypeTy::SUBMIT:
    return "Submit";
  case EventTypeTy::EXCHANGE:
    return "Exchange";
  case EventTypeTy::EXECUTE:
    return "Execute";
  case EventTypeTy::SYNC:
    return "Sync";
  case EventTypeTy::EXIT:
    return "Exit";
  }

  assertm(false, "Every enum value must be checked on the switch above.");
  return nullptr;
}

const char *toString(EventLocationTy Location) {
  switch (Location) {
  case EventLocationTy::DEST:
    return "Destination";
  case EventLocationTy::ORIG:
    return "Origin";
  }

  assertm(false, "Every enum value must be checked on the switch above.");
  return nullptr;
}

const char *toString(EventStateTy State) {
  switch (State) {
  case EventStateTy::CREATED:
    return "Created";
  case EventStateTy::EXECUTING:
    return "Executing";
  case EventStateTy::WAITING:
    return "Waiting";
  case EventStateTy::FAILED:
    return "Failed";
  case EventStateTy::FINISHED:
    return "Finished";
  }

  assertm(false, "Every enum value must be checked on the switch above.");
  return nullptr;
}

// Base Event implementation
// =============================================================================
BaseEventTy::BaseEventTy(EventLocationTy EventLocation, EventTypeTy EventType,
                         int MPITag, MPI_Comm TargetComm, int OrigRank,
                         int DestRank)
    : EventLocation(EventLocation), EventType(EventType),
      TargetComm(TargetComm), MPITag(MPITag), OrigRank(OrigRank),
      DestRank(DestRank) {
  assertm(MPITag >= static_cast<int>(ControlTagsTy::FIRST_EVENT),
          "Event MPI tag must not have a Control Tag value");
  assertm(MPITag <= EventSystemTy::MPITagMaxValue,
          "Event MPI tag must be smaller than the maximum value allowed");
}

void BaseEventTy::notifyNewEvent() {
  if (EventLocation == EventLocationTy::ORIG) {
    // Sends event request.
    InitialRequestInfo[0] = static_cast<uint32_t>(EventType);
    InitialRequestInfo[1] = static_cast<uint32_t>(MPITag);

    if (OrigRank != DestRank)
      MPI_Isend(InitialRequestInfo, 2, MPI_UINT32_T, DestRank,
                static_cast<int>(ControlTagsTy::EVENT_REQUEST),
                EventSystemTy::GateThreadComm, getNextRequest());
  }
}

bool BaseEventTy::runCoroutine() {
  if (EventLocation == EventLocationTy::ORIG) {
    return runOrigin();
  } else {
    return runDestination();
  }
}

bool BaseEventTy::checkPendingRequests() {
  int RequestsCompleted = false;

  MPI_Testall(PendingRequests.size(), PendingRequests.data(),
              &RequestsCompleted, MPI_STATUSES_IGNORE);

  return RequestsCompleted;
}

bool BaseEventTy::isDone() const {
  return (EventState == EventStateTy::FINISHED) ||
         (EventState == EventStateTy::FAILED);
}

void BaseEventTy::progress() {
  // Immediately return if the event is already finished
  if (isDone()) {
    return;
  }

  // The following code block uses a guard to ensure only one thread is
  // executing the progress function at a time, returning immediately for all
  // the other threads (e.g. multiple threads waiting on the same event).
  //
  // If one thread is already advancing the event execution at a node, there is
  // no need for other threads to execute the progress function. Returning
  // immediately frees other threads to execute other events/procedures and
  // allows the events run* coroutines to be implemented in a not thead-safe
  // manner.
  bool ExpectedProgressGuard = false;
  if (!ProgressGuard.compare_exchange_weak(ExpectedProgressGuard, true)) {
    return;
  }

  // Advance the event local execution depending on its state.
  switch (EventState) {
  case EventStateTy::CREATED:
    notifyNewEvent();
    EventState = EventStateTy::EXECUTING;
    [[fallthrough]];

  case EventStateTy::EXECUTING:
    if (!runCoroutine())
      break;
    EventState = EventStateTy::WAITING;
    [[fallthrough]];

  case EventStateTy::WAITING:
    if (!checkPendingRequests())
      break;
    if (EventState != EventStateTy::FAILED)
      EventState = EventStateTy::FINISHED;
    break;

  case EventStateTy::FAILED:
    REPORT("MPI event %s failed.\n", toString(EventType));
    [[fallthrough]];

  case EventStateTy::FINISHED:
    break;
  }

  // Allow other threads to call progress again.
  ProgressGuard = false;
}

void BaseEventTy::wait() {
  // Advance the event progress until it is completed.
  while (!isDone()) {
    progress();

    std::this_thread::sleep_for(
        std::chrono::microseconds(config::EVENT_POLLING_RATE));
  }
}

EventStateTy BaseEventTy::getEventState() const { return EventState; }

MPI_Request *BaseEventTy::getNextRequest() {
  PendingRequests.emplace_back(MPI_REQUEST_NULL);
  return &PendingRequests.back();
}

// Alloc Event implementation
// =============================================================================
AllocEventTy::AllocEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank,
                           int DestRank, int64_t Size, void **AllocatedAddress)
    : BaseEventTy(EventLocationTy::ORIG, EventTypeTy::ALLOC, MPITag, TargetComm,
                  OrigRank, DestRank),
      Size(Size), AllocatedAddressPtr(AllocatedAddress) {
  assertm(Size >= 0, "AllocEvent must receive a Size >= 0");
  assertm(AllocatedAddress != nullptr,
          "AllocEvent must receive a valid pointer as AllocatedAddress");
}

bool AllocEventTy::runOrigin() {
  assert(EventLocation == EventLocationTy::ORIG);

  EVENT_BEGIN();

  MPI_Isend(&Size, 1, MPI_INT64_T, DestRank, MPITag, TargetComm,
            getNextRequest());

  MPI_Irecv(AllocatedAddressPtr, sizeof(uintptr_t), MPI_BYTE, DestRank, MPITag,
            TargetComm, getNextRequest());

  EVENT_END();
}

AllocEventTy::AllocEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank,
                           int DestRank)
    : BaseEventTy(EventLocationTy::DEST, EventTypeTy::ALLOC, MPITag, TargetComm,
                  OrigRank, DestRank) {}

bool AllocEventTy::runDestination() {
  assert(EventLocation == EventLocationTy::DEST);

  EVENT_BEGIN();

  MPI_Irecv(&Size, 1, MPI_INT64_T, OrigRank, MPITag, TargetComm,
            getNextRequest());

  EVENT_PAUSE_FOR_REQUESTS();

  DestAddress = malloc(Size);

  MPI_Isend(&DestAddress, sizeof(uintptr_t), MPI_BYTE, OrigRank, MPITag,
            TargetComm, getNextRequest());

  EVENT_END();
}

// Delete Event implementation
// =============================================================================
DeleteEventTy::DeleteEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank,
                             int DestRank, void *TargetAddress)
    : BaseEventTy(EventLocationTy::ORIG, EventTypeTy::DELETE, MPITag,
                  TargetComm, OrigRank, DestRank),
      TargetAddress(TargetAddress) {}

bool DeleteEventTy::runOrigin() {
  assert(EventLocation == EventLocationTy::ORIG);

  EVENT_BEGIN();

  MPI_Isend(&TargetAddress, sizeof(void *), MPI_BYTE, DestRank, MPITag,
            TargetComm, getNextRequest());

  // Event completion notification
  MPI_Irecv(nullptr, 0, MPI_BYTE, DestRank, MPITag, TargetComm,
            getNextRequest());

  EVENT_END();
}

DeleteEventTy::DeleteEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank,
                             int DestRank)
    : BaseEventTy(EventLocationTy::DEST, EventTypeTy::DELETE, MPITag,
                  TargetComm, OrigRank, DestRank) {}

bool DeleteEventTy::runDestination() {
  assert(EventLocation == EventLocationTy::DEST);

  EVENT_BEGIN();

  MPI_Irecv(&TargetAddress, sizeof(void *), MPI_BYTE, OrigRank, MPITag,
            TargetComm, getNextRequest());

  EVENT_PAUSE_FOR_REQUESTS();

  free(TargetAddress);

  // Event completion notification
  MPI_Isend(nullptr, 0, MPI_BYTE, OrigRank, MPITag, TargetComm,
            getNextRequest());

  EVENT_END();
}

// Retrieve Event implementation
// =============================================================================
RetrieveEventTy::RetrieveEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank,
                                 int DestRank, void *OrigPtr,
                                 const void *DestPtr, int64_t Size)
    : BaseEventTy(EventLocationTy::ORIG, EventTypeTy::RETRIEVE, MPITag,
                  TargetComm, OrigRank, DestRank),
      OrigPtr(OrigPtr), DestPtr(DestPtr), Size(Size) {
  assertm(Size >= 0, "RetrieveEvent must receive a Size >= 0");
  assertm(OrigPtr != nullptr,
          "RetrieveEvent must receive a valid pointer as OrigPtr");
  assertm(DestPtr != nullptr,
          "RetrieveEvent must receive a valid pointer as DestPtr");
}

bool RetrieveEventTy::runOrigin() {
  assert(EventLocation == EventLocationTy::ORIG);

  char *BufferByteArray = nullptr;
  int64_t RemainingBytes = 0;

  EVENT_BEGIN();

  MPI_Isend(&DestPtr, sizeof(uintptr_t), MPI_BYTE, DestRank, MPITag, TargetComm,
            getNextRequest());

  MPI_Isend(&Size, sizeof(int64_t), MPI_BYTE, DestRank, MPITag, TargetComm,
            getNextRequest());

  // TODO: Extract this to an common function for both dest/orig for
  // submit/retrieve.
  // Operates over many fragments of the original buffer of at
  // most MPI_FRAGMENT_SIZE bytes.
  BufferByteArray = reinterpret_cast<char *>(OrigPtr);
  RemainingBytes = Size;
  while (RemainingBytes > 0) {
    MPI_Irecv(
        &BufferByteArray[Size - RemainingBytes],
        static_cast<int>(std::min(RemainingBytes, config::MPI_FRAGMENT_SIZE)),
        MPI_BYTE, DestRank, MPITag, TargetComm, getNextRequest());
    RemainingBytes -= config::MPI_FRAGMENT_SIZE;
  }

  EVENT_END();
}

RetrieveEventTy::RetrieveEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank,
                                 int DestRank)
    : BaseEventTy(EventLocationTy::DEST, EventTypeTy::RETRIEVE, MPITag,
                  TargetComm, OrigRank, DestRank) {}

bool RetrieveEventTy::runDestination() {
  assert(EventLocation == EventLocationTy::DEST);

  const char *BufferByteArray = nullptr;
  int64_t RemainingBytes = 0;

  EVENT_BEGIN();

  MPI_Irecv(&DestPtr, sizeof(uintptr_t), MPI_BYTE, OrigRank, MPITag, TargetComm,
            getNextRequest());

  MPI_Irecv(&Size, sizeof(int64_t), MPI_BYTE, OrigRank, MPITag, TargetComm,
            getNextRequest());

  EVENT_PAUSE_FOR_REQUESTS();

  // Operates over many fragments of the original buffer of at most
  // MPI_FRAGMENT_SIZE bytes.
  BufferByteArray = reinterpret_cast<const char *>(DestPtr);
  RemainingBytes = Size;
  while (RemainingBytes > 0) {
    MPI_Isend(
        &BufferByteArray[Size - RemainingBytes],
        static_cast<int>(std::min(RemainingBytes, config::MPI_FRAGMENT_SIZE)),
        MPI_BYTE, OrigRank, MPITag, TargetComm, getNextRequest());
    RemainingBytes -= config::MPI_FRAGMENT_SIZE;
  }

  EVENT_END();
}

// Submit Event implementation
// =============================================================================
SubmitEventTy::SubmitEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank,
                             int DestRank, const void *OrigPtr, void *DestPtr,
                             int64_t Size)
    : BaseEventTy(EventLocationTy::ORIG, EventTypeTy::SUBMIT, MPITag,
                  TargetComm, OrigRank, DestRank),
      OrigPtr(OrigPtr), DestPtr(DestPtr), Size(Size) {
  assertm(Size >= 0, "SubmitEvent must receive a Size >= 0");
  assertm(OrigPtr != nullptr,
          "SubmitEvent must receive a valid pointer as OrigPtr");
  assertm(DestPtr != nullptr,
          "SubmitEvent must receive a valid pointer as DestPtr");
}

bool SubmitEventTy::runOrigin() {
  assert(EventLocation == EventLocationTy::ORIG);

  const char *BufferByteArray;
  int64_t RemainingBytes;

  EVENT_BEGIN();

  MPI_Isend(&DestPtr, sizeof(uintptr_t), MPI_BYTE, DestRank, MPITag, TargetComm,
            getNextRequest());

  MPI_Isend(&Size, sizeof(int64_t), MPI_BYTE, DestRank, MPITag, TargetComm,
            getNextRequest());

  // Operates over many fragments of the original buffer of at most
  // MPI_FRAGMENT_SIZE bytes.
  BufferByteArray = reinterpret_cast<const char *>(OrigPtr);
  RemainingBytes = Size;
  while (RemainingBytes > 0) {
    MPI_Isend(
        &BufferByteArray[Size - RemainingBytes],
        static_cast<int>(std::min(RemainingBytes, config::MPI_FRAGMENT_SIZE)),
        MPI_BYTE, DestRank, MPITag, TargetComm, getNextRequest());
    RemainingBytes -= config::MPI_FRAGMENT_SIZE;
  }

  // Event completion notification
  MPI_Irecv(nullptr, 0, MPI_BYTE, DestRank, MPITag, TargetComm,
            getNextRequest());

  EVENT_END();
}

SubmitEventTy::SubmitEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank,
                             int DestRank)
    : BaseEventTy(EventLocationTy::DEST, EventTypeTy::SUBMIT, MPITag,
                  TargetComm, OrigRank, DestRank) {}

bool SubmitEventTy::runDestination() {
  assert(EventLocation == EventLocationTy::DEST);

  char *BufferByteArray = nullptr;
  int64_t RemainingBytes = 0;

  EVENT_BEGIN();

  MPI_Irecv(&DestPtr, sizeof(uintptr_t), MPI_BYTE, OrigRank, MPITag, TargetComm,
            getNextRequest());

  MPI_Irecv(&Size, sizeof(int64_t), MPI_BYTE, OrigRank, MPITag, TargetComm,
            getNextRequest());

  EVENT_PAUSE_FOR_REQUESTS();

  // Operates over many fragments of the original buffer of at most
  // MPI_FRAGMENT_SIZE bytes.
  BufferByteArray = reinterpret_cast<char *>(DestPtr);
  RemainingBytes = Size;
  while (RemainingBytes > 0) {
    MPI_Irecv(
        &BufferByteArray[Size - RemainingBytes],
        static_cast<int>(std::min(RemainingBytes, config::MPI_FRAGMENT_SIZE)),
        MPI_BYTE, OrigRank, MPITag, TargetComm, getNextRequest());
    RemainingBytes -= config::MPI_FRAGMENT_SIZE;
  }

  // Event completion notification
  MPI_Isend(nullptr, 0, MPI_BYTE, OrigRank, MPITag, TargetComm,
            getNextRequest());

  EVENT_END();
}

// Exchange Event implementation
// =============================================================================
ExchangeEventTy::ExchangeEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank,
                                 int DestRank, int data_dst_rank,
                                 const void *src_ptr, void *dst_ptr,
                                 int64_t Size)
    : BaseEventTy(EventLocationTy::ORIG, EventTypeTy::EXCHANGE, MPITag,
                  TargetComm, OrigRank, DestRank),
      DataDestRank(data_dst_rank), SrcPtr(src_ptr), DstPtr(dst_ptr),
      Size(Size) {
  assertm(Size >= 0, "ExchangeEvent must receive a Size >= 0");
  assertm(src_ptr != nullptr,
          "ExchangeEvent must receive a valid pointer as src_ptr");
  assertm(dst_ptr != nullptr,
          "ExchangeEvent must receive a valid pointer as dst_ptr");
}

bool ExchangeEventTy::runOrigin() {
  assert(EventLocation == EventLocationTy::ORIG);

  EVENT_BEGIN();

  MPI_Isend(&DataDestRank, sizeof(int), MPI_BYTE, DestRank, MPITag, TargetComm,
            getNextRequest());

  MPI_Isend(&SrcPtr, sizeof(uintptr_t), MPI_BYTE, DestRank, MPITag, TargetComm,
            getNextRequest());

  MPI_Isend(&DstPtr, sizeof(uintptr_t), MPI_BYTE, DestRank, MPITag, TargetComm,
            getNextRequest());

  MPI_Isend(&Size, sizeof(int64_t), MPI_BYTE, DestRank, MPITag, TargetComm,
            getNextRequest());

  // Event completion notification
  MPI_Irecv(nullptr, 0, MPI_BYTE, DestRank, MPITag, TargetComm,
            getNextRequest());

  EVENT_END();
}

ExchangeEventTy::ExchangeEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank,
                                 int DestRank)
    : BaseEventTy(EventLocationTy::DEST, EventTypeTy::EXCHANGE, MPITag,
                  TargetComm, OrigRank, DestRank),
      DataDestRank(-1), SrcPtr(nullptr), DstPtr(nullptr), Size(0) {}

bool ExchangeEventTy::runDestination() {
  assert(EventLocation == EventLocationTy::DEST);

  EVENT_BEGIN();

  MPI_Irecv(&DataDestRank, sizeof(int), MPI_BYTE, OrigRank, MPITag, TargetComm,
            getNextRequest());

  MPI_Irecv(&SrcPtr, sizeof(uintptr_t), MPI_BYTE, OrigRank, MPITag, TargetComm,
            getNextRequest());

  MPI_Irecv(&DstPtr, sizeof(uintptr_t), MPI_BYTE, OrigRank, MPITag, TargetComm,
            getNextRequest());

  MPI_Irecv(&Size, sizeof(int64_t), MPI_BYTE, OrigRank, MPITag, TargetComm,
            getNextRequest());

  EVENT_PAUSE_FOR_REQUESTS();

  RemoteSubmitEvent = std::make_shared<SubmitEventTy>(
      MPITag, TargetComm, DestRank, DataDestRank, SrcPtr, DstPtr, Size);

  do {
    RemoteSubmitEvent->progress();

    if (!RemoteSubmitEvent->isDone()) {
      EVENT_PAUSE();
    }
  } while (!RemoteSubmitEvent->isDone());

  // Event completion notification
  MPI_Isend(nullptr, 0, MPI_BYTE, OrigRank, MPITag, TargetComm,
            getNextRequest());

  EVENT_END();
}

// Execute Event implementation
// =============================================================================
ExecuteEventTy::ExecuteEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank,
                               int DestRank, int32_t NumArgs, void **ArgsArray,
                               uint32_t TargetEntryIdx)
    : BaseEventTy(EventLocationTy::ORIG, EventTypeTy::EXECUTE, MPITag,
                  TargetComm, OrigRank, DestRank),
      NumArgs(NumArgs), Args(NumArgs, nullptr), TargetEntryIdx(TargetEntryIdx) {
  assertm(NumArgs >= 0, "ExecuteEvent must receive an NumArgs >= 0");
  assertm(NumArgs == 0 || ArgsArray != nullptr,
          "ExecuteEvent must receive a valid Args when NumArgs > 0");

  std::copy_n(ArgsArray, NumArgs, Args.begin());
}

bool ExecuteEventTy::runOrigin() {
  assert(EventLocation == EventLocationTy::ORIG);

  EVENT_BEGIN();

  MPI_Isend(&NumArgs, sizeof(int32_t), MPI_BYTE, DestRank, MPITag, TargetComm,
            getNextRequest());

  MPI_Isend(Args.data(), NumArgs * sizeof(uintptr_t), MPI_BYTE, DestRank,
            MPITag, TargetComm, getNextRequest());

  MPI_Isend(&TargetEntryIdx, sizeof(uint32_t), MPI_BYTE, DestRank, MPITag,
            TargetComm, getNextRequest());

  // Event completion notification
  MPI_Irecv(nullptr, 0, MPI_BYTE, DestRank, MPITag, TargetComm,
            getNextRequest());

  EVENT_END();
}

ExecuteEventTy::ExecuteEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank,
                               int DestRank, __tgt_target_table *TargetTable)
    : BaseEventTy(EventLocationTy::DEST, EventTypeTy::EXECUTE, MPITag,
                  TargetComm, OrigRank, DestRank),
      TargetTable(TargetTable) {
  assertm(TargetTable != nullptr,
          "ExecuteEvent must receive a valid pointer as TargetTable");
}

bool ExecuteEventTy::runDestination() {
  assert(EventLocation == EventLocationTy::DEST);

  __tgt_offload_entry *Begin = nullptr;
  __tgt_offload_entry *End = nullptr;
  __tgt_offload_entry *Curr = nullptr;
  ffi_cif Cif{};
  llvm::SmallVector<ffi_type *> ArgsTypes{};
  ffi_status FFIStatus [[maybe_unused]] = FFI_OK;
  void (*TargetEntry)(void) = nullptr;

  EVENT_BEGIN();

  MPI_Irecv(&NumArgs, sizeof(int32_t), MPI_BYTE, OrigRank, MPITag, TargetComm,
            getNextRequest());

  EVENT_PAUSE_FOR_REQUESTS();

  Args.resize(NumArgs, nullptr);
  ArgsTypes.resize(NumArgs, &ffi_type_pointer);
  MPI_Irecv(Args.data(), NumArgs * sizeof(uintptr_t), MPI_BYTE, OrigRank,
            MPITag, TargetComm, getNextRequest());

  MPI_Irecv(&TargetEntryIdx, sizeof(uint32_t), MPI_BYTE, OrigRank, MPITag,
            TargetComm, getNextRequest());

  EVENT_PAUSE_FOR_REQUESTS();

  // Iterates over all the host table entries to see if we can locate the
  // host_ptr.
  Begin = TargetTable->EntriesBegin;
  End = TargetTable->EntriesEnd;
  Curr = Begin;

  // Iterates over all the table entries to see if we can locate the entry.
  for (uint32_t I = 0; Curr < End; ++Curr, ++I) {
    if (I == TargetEntryIdx) {
      // We got a match, now fill the HostPtrToTableMap so that we may avoid
      // this search next time.
      *((void **)&TargetEntry) = Curr->addr;
      break;
    }
  }

  // Return failure when entry not found.
  assertm(Curr != End, "Could not find the right entry");

  FFIStatus = ffi_prep_cif(&Cif, FFI_DEFAULT_ABI, NumArgs, &ffi_type_void,
                           &ArgsTypes[0]);

  assertm(FFIStatus == FFI_OK, "Unable to prepare target launch!");

  ffi_call(&Cif, TargetEntry, NULL, &Args[0]);

  // Event completion notification
  MPI_Isend(nullptr, 0, MPI_BYTE, OrigRank, MPITag, TargetComm,
            getNextRequest());

  EVENT_END();
}

// Sync Event implementation
// =============================================================================
SyncEventTy::SyncEventTy(EventPtr &target_event)
    : BaseEventTy(EventLocationTy::DEST, EventTypeTy::SYNC,
                  EventSystemTy::MPITagMaxValue, 0, 0, 0),
      TargetEvent(target_event) {}

bool SyncEventTy::runOrigin() { return true; }

bool SyncEventTy::runDestination() {
  EVENT_BEGIN();

  while (!TargetEvent->isDone()) {
    EVENT_PAUSE();
  }

  EVENT_END();
}

// Exit Event implementation
// =============================================================================
ExitEventTy::ExitEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank,
                         int DestRank)
    : BaseEventTy(EventLocationTy::ORIG, EventTypeTy::EXIT, MPITag, TargetComm,
                  OrigRank, DestRank) {}

bool ExitEventTy::runOrigin() {
  assert(EventLocation == EventLocationTy::ORIG);

  EVENT_BEGIN();

  // Event completion notification
  MPI_Irecv(nullptr, 0, MPI_BYTE, DestRank, MPITag, TargetComm,
            getNextRequest());

  EVENT_END();
}

ExitEventTy::ExitEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank,
                         int DestRank,
                         std::atomic<EventSystemStateTy> *EventSystemState)
    : BaseEventTy(EventLocationTy::DEST, EventTypeTy::EXIT, MPITag, TargetComm,
                  OrigRank, DestRank),
      EventSystemState(EventSystemState) {}

bool ExitEventTy::runDestination() {
  assert(EventLocation == EventLocationTy::DEST);

  EventSystemStateTy OldState;

  EVENT_BEGIN();

  OldState = EventSystemState->exchange(EventSystemStateTy::EXITED);
  assertm(OldState != EventSystemStateTy::EXITED,
          "Exit event received multiple times");

  // Event completion notification
  MPI_Isend(nullptr, 0, MPI_BYTE, OrigRank, MPITag, TargetComm,
            getNextRequest());

  EVENT_END();
}

// Event Queue implementation
// =============================================================================
EventQueue::EventQueue() : Queue(), QueueMtx(), CanPopCv() {}

size_t EventQueue::size() {
  std::lock_guard<std::mutex> lock(QueueMtx);
  return Queue.size();
}

void EventQueue::push(EventPtr &Event) {
  {
    std::unique_lock<std::mutex> lock(QueueMtx);
    Queue.push(Event);
  }

  // Notifies a thread possibly blocked by an empty queue.
  CanPopCv.notify_one();
}

EventPtr EventQueue::pop() {
  EventPtr TargetEvent = nullptr;

  {
    std::unique_lock<std::mutex> lock(QueueMtx);

    // Waits for at least one item to be pushed.
    while (Queue.empty()) {
      const bool has_new_event = CanPopCv.wait_for(
          lock, std::chrono::microseconds(config::EVENT_POLLING_RATE),
          [&] { return !Queue.empty(); });

      if (!has_new_event) {
        return nullptr;
      }
    }

    assertm(!Queue.empty(), "Queue was empty on pop operation.");

    TargetEvent = Queue.front();
    Queue.pop();
  }

  return TargetEvent;
}

// Event System implementation
// =============================================================================
// Event System statics.
MPI_Comm EventSystemTy::GateThreadComm = MPI_COMM_NULL;
int32_t EventSystemTy::MPITagMaxValue = 0;

EventSystemTy::EventSystemTy() : EventSystemState(EventSystemStateTy::CREATED) {
  // Read environment parameters
  if (const char *env_str = std::getenv("OMPCLUSTER_MPI_FRAGMENT_SIZE")) {
    config::MPI_FRAGMENT_SIZE = std::stoi(env_str);
    assertm(config::MPI_FRAGMENT_SIZE >= 1,
            "Maximum MPI buffer Size must be a least 1");
    assertm(config::MPI_FRAGMENT_SIZE < std::numeric_limits<int>::max(),
            "Maximum MPI buffer Size must be less then the largest int "
            "value (MPI restrictions)");
  }

  if (const char *env_str = std::getenv("OMPCLUSTER_NUM_EXEC_EVENT_HANDLERS")) {
    config::NUM_EXEC_EVENT_HANDLERS = std::stoi(env_str);
    assertm(config::NUM_EXEC_EVENT_HANDLERS >= 1,
            "At least one exec event handler should be spawned");
  }

  if (const char *env_str = std::getenv("OMPCLUSTER_NUM_DataEventHandlers")) {
    config::NUM_DATA_EVENT_HANDLERS = std::stoi(env_str);
    assertm(config::NUM_DATA_EVENT_HANDLERS >= 1,
            "At least one data event handler should be spawned");
  }

  if (const char *env_str = std::getenv("OMPCLUSTER_EVENT_POLLING_RATE")) {
    config::EVENT_POLLING_RATE = std::stoi(env_str);
    assertm(config::EVENT_POLLING_RATE >= 0,
            "Event system polling rate should not be negative");
  }

  if (const char *env_str = std::getenv("OMPCLUSTER_NUM_EVENT_COMM")) {
    config::NUM_EVENT_COMM = std::stoi(env_str);
    assertm(config::NUM_EVENT_COMM >= 1,
            "At least on communicator need to be spawned");
  }
}

EventSystemTy::~EventSystemTy() {
  if (!IsInitialized)
    return;

  REPORT("Destructing internal event system before deinitializing it.\n");
  deinitialize();
}

bool EventSystemTy::initialize() {
  if (IsInitialized) {
    REPORT("Trying to initialize event system twice.\n");
    return false;
  }

  if (!createLocalMPIContext())
    return false;

  IsInitialized = true;

  return true;
}

bool EventSystemTy::deinitialize() {
  if (!IsInitialized) {
    REPORT("Trying to deinitialize event system twice.\n");
    return false;
  }

  // Only send exit events from the host side
  if (isHead() && WorldSize > 1) {
    const int NumWorkers = WorldSize - 1;
    llvm::SmallVector<EventPtr> ExitEvents(NumWorkers);
    for (int WorkerRank = 0; WorkerRank < NumWorkers; WorkerRank++) {
      ExitEvents[WorkerRank] = createEvent<ExitEventTy>(WorkerRank);
      ExitEvents[WorkerRank]->progress();
    }

    bool SuccessfullyExited = true;
    for (int WorkerRank = 0; WorkerRank < NumWorkers; WorkerRank++) {
      ExitEvents[WorkerRank]->wait();
      SuccessfullyExited &=
          ExitEvents[WorkerRank]->getEventState() == EventStateTy::FINISHED;
    }

    if (!SuccessfullyExited) {
      REPORT("Failed to stop worker processes.\n");
      return false;
    }
  }

  if (!destroyLocalMPIContext())
    return false;

  IsInitialized = false;

  return true;
}

void EventSystemTy::runEventHandler(EventQueue &Queue) {
  while (EventSystemState == EventSystemStateTy::RUNNING || Queue.size() > 0) {
    EventPtr event = Queue.pop();

    // Re-checks the stop condition when no event was found.
    if (event == nullptr) {
      continue;
    }

    event->progress();

    if (!event->isDone()) {
      Queue.push(event);
    }
  }
}

void EventSystemTy::runGateThread(__tgt_target_table *TargetTable) {
  // Updates the event state and
  EventSystemState = EventSystemStateTy::RUNNING;

  // Spawns the event handlers.
  llvm::SmallVector<std::thread> EventHandlers;
  EventHandlers.resize(config::NUM_EXEC_EVENT_HANDLERS +
                       config::NUM_DATA_EVENT_HANDLERS);
  for (int Idx = 0; Idx < EventHandlers.size(); Idx++) {
    EventHandlers[Idx] = std::thread(
        &EventSystemTy::runEventHandler, this,
        std::ref(Idx < config::NUM_EXEC_EVENT_HANDLERS ? ExecEventQueue
                                                       : DataEventQueue));
  }

  // Executes the gate thread logic
  while (EventSystemState == EventSystemStateTy::RUNNING) {
    // Checks for new incoming event requests.
    MPI_Message EventReqMsg;
    MPI_Status EventStatus;
    int HasReceived = false;
    MPI_Improbe(MPI_ANY_SOURCE, static_cast<int>(ControlTagsTy::EVENT_REQUEST),
                GateThreadComm, &HasReceived, &EventReqMsg, MPI_STATUS_IGNORE);

    // If none was received, wait for `EVENT_POLLING_RATE`us for the next
    // check.
    if (!HasReceived) {
      std::this_thread::sleep_for(
          std::chrono::microseconds(config::EVENT_POLLING_RATE));
      continue;
    }

    // Acquires the event information from the received request, which are:
    // - Event type
    // - Event tag
    // - Target comm
    // - Event source rank
    uint32_t EventInfo[2];
    MPI_Mrecv(EventInfo, 2, MPI_UINT32_T, &EventReqMsg, &EventStatus);
    const auto NewEventType = static_cast<EventTypeTy>(EventInfo[0]);
    const uint32_t NewEventTag = EventInfo[1];
    auto &NewEventComm = getNewEventComm(NewEventTag);
    const int OrigRank = EventStatus.MPI_SOURCE;

    // Creates a new receive event of 'event_type' type.
    EventPtr NewEvent;
    switch (NewEventType) {
    case EventTypeTy::ALLOC:
      NewEvent = std::make_shared<AllocEventTy>(NewEventTag, NewEventComm,
                                                OrigRank, LocalRank);
      break;
    case EventTypeTy::DELETE:
      NewEvent = std::make_shared<DeleteEventTy>(NewEventTag, NewEventComm,
                                                 OrigRank, LocalRank);
      break;
    case EventTypeTy::RETRIEVE:
      NewEvent = std::make_shared<RetrieveEventTy>(NewEventTag, NewEventComm,
                                                   OrigRank, LocalRank);
      break;
    case EventTypeTy::SUBMIT:
      NewEvent = std::make_shared<SubmitEventTy>(NewEventTag, NewEventComm,
                                                 OrigRank, LocalRank);
      break;
    case EventTypeTy::EXCHANGE:
      NewEvent = std::make_shared<ExchangeEventTy>(NewEventTag, NewEventComm,
                                                   OrigRank, LocalRank);
      break;
    case EventTypeTy::EXECUTE:
      NewEvent = std::make_shared<ExecuteEventTy>(
          NewEventTag, NewEventComm, OrigRank, LocalRank, TargetTable);
      break;
    case EventTypeTy::EXIT:
      NewEvent = std::make_shared<ExitEventTy>(
          NewEventTag, NewEventComm, OrigRank, LocalRank, &EventSystemState);
      break;
    case EventTypeTy::SYNC:
      assertm(false, "Trying to create a local event on a remote node");
    }

    assertm(NewEvent != nullptr, "Created event must not be a nullptr");
    assertm(NewEvent->EventLocation == EventLocationTy::DEST,
            "Gate thread must receive only receive events");

    if (NewEventType == EventTypeTy::EXECUTE) {
      ExecEventQueue.push(NewEvent);
    } else {
      DataEventQueue.push(NewEvent);
    }
  }

  assertm(EventSystemState == EventSystemStateTy::EXITED,
          "Event State should be EXITED after receiving an Exit event");

  // Waits for the Event Handler threads.
  for (auto &EventHandler : EventHandlers) {
    if (EventHandler.joinable()) {
      EventHandler.join();
    } else {
      assertm(false, "Event Handler threads not joinable at the end of gate "
                     "thread logic.");
    }
  }
}

// Creates a new event tag of at least 'FIRST_EVENT' value.
// Tag values smaller than 'FIRST_EVENT' are reserved for control
// communication between the event systems of different MPI processes.
int EventSystemTy::createNewEventTag() {
  uint32_t tag = 0;

  do {
    tag = EventCounter.fetch_add(1) % MPITagMaxValue;
  } while (tag < static_cast<int>(ControlTagsTy::FIRST_EVENT));

  return tag;
}

MPI_Comm &EventSystemTy::getNewEventComm(int MPITag) {
  // Retrieve a comm using a round-robin strategy around the event's mpi tag.
  return EventCommPool[MPITag % EventCommPool.size()];
}

static const char *threadLevelToString(int ThreadLevel) {
  switch (ThreadLevel) {
  case MPI_THREAD_SINGLE:
    return "MPI_THREAD_SINGLE";
  case MPI_THREAD_SERIALIZED:
    return "MPI_THREAD_SERIALIZED";
  case MPI_THREAD_FUNNELED:
    return "MPI_THREAD_FUNNELED";
  case MPI_THREAD_MULTIPLE:
    return "MPI_THREAD_MULTIPLE";
  default:
    return "unkown";
  }
}

bool EventSystemTy::createLocalMPIContext() {
  int MPIError = MPI_SUCCESS;

  // Initialize the MPI context.
  int IsMPIInitialized = 0;
  int ThreadLevel = MPI_THREAD_SINGLE;

  MPI_Initialized(&IsMPIInitialized);

  if (IsMPIInitialized)
    MPI_Query_thread(&ThreadLevel);
  else
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &ThreadLevel);

  CHECK(ThreadLevel == MPI_THREAD_MULTIPLE,
        "MPI plugin requires a MPI implementation with %s thread level. "
        "Implementation only supports up to %s.\n",
        threadLevelToString(MPI_THREAD_MULTIPLE),
        threadLevelToString(ThreadLevel));

  // Create gate thread comm.
  MPIError = MPI_Comm_dup(MPI_COMM_WORLD, &GateThreadComm);
  CHECK(MPIError == MPI_SUCCESS,
        "Failed to create gate thread MPI comm with error %d\n", MPIError);

  // Create event comm pool.
  EventCommPool.resize(config::NUM_EVENT_COMM, MPI_COMM_NULL);
  for (auto &Comm : EventCommPool) {
    MPI_Comm_dup(MPI_COMM_WORLD, &Comm);
    CHECK(MPIError == MPI_SUCCESS,
          "Failed to create MPI comm pool with error %d\n", MPIError);
  }

  // Get local MPI process description.
  MPIError = MPI_Comm_rank(GateThreadComm, &LocalRank);
  CHECK(MPIError == MPI_SUCCESS,
        "Failed to acquire the local MPI rank with error %d\n", MPIError);

  MPIError = MPI_Comm_size(GateThreadComm, &WorldSize);
  CHECK(MPIError == MPI_SUCCESS,
        "Failed to acquire the world size with error %d\n", MPIError);

  // Get max value for MPI tags.
  MPI_Aint *Value = nullptr;
  int Flag = 0;
  MPIError = MPI_Comm_get_attr(GateThreadComm, MPI_TAG_UB, &Value, &Flag);
  CHECK(Flag == 1 && MPIError == MPI_SUCCESS,
        "Failed to acquire the MPI max tag value with error %d\n", MPIError);
  MPITagMaxValue = *Value;

  return true;
}

bool EventSystemTy::destroyLocalMPIContext() {
  int MPIError = MPI_SUCCESS;

  // Note: We don't need to assert here since application part of the program
  // was finished.
  // Free gate thread comm.
  MPIError = MPI_Comm_free(&GateThreadComm);
  CHECK(MPIError == MPI_SUCCESS,
        "Failed to destroy the gate thread MPI comm with error %d\n", MPIError);

  // Free event comm pool.
  for (auto &comm : EventCommPool) {
    MPI_Comm_free(&comm);
    CHECK(MPIError == MPI_SUCCESS,
          "Failed to destroy the event MPI comm with error %d\n", MPIError);
  }
  EventCommPool.resize(0);

  // Finalize the global MPI session.
  int IsFinalized = false;
  MPIError = MPI_Finalized(&IsFinalized);

  if (IsFinalized) {
    DP("MPI was already finalized externally.\n");
  } else {
    MPIError = MPI_Finalize();
    CHECK(MPIError == MPI_SUCCESS, "Failed to finalize MPI with error: %d\n",
          MPIError);
  }

  return true;
}

int EventSystemTy::getNumWorkers() const { return WorldSize - 1; };

int EventSystemTy::isHead() const { return LocalRank == 0; };
