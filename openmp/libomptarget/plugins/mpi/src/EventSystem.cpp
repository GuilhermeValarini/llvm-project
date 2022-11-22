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
#include <limits>
#include <sstream>

#include <ffi.h>

#include "Debug.h"
#include "Utilities.h"
#include "omptarget.h"

#define CHECK(expr, msg, ...)                                                  \
  if (!(expr)) {                                                               \
    REPORT(msg, ##__VA_ARGS__);                                                \
    return false;                                                              \
  }

// Customizable parameters of the event system
// =============================================================================
// Number of execute event handlers to spawn.
static llvm::omp::target::IntEnvar
    NumExecEventHandlers("OMPTARGET_NUM_EXEC_EVENT_HANDLERS", 100e6);
// Number of data event handlers to spawn.
static llvm::omp::target::IntEnvar
    NumDataEventHandlers("OMPTARGET_NUM_DATA_EVENT_HANDLERS", 1);
// Polling rate period (us) used by event handlers.
static llvm::omp::target::IntEnvar
    EventPollingRate("OMPTARGET_EVENT_POLLING_RATE", 1);
// Number of communicators to be spawned and distributed for the events.
// Allows for parallel use of network resources.
static llvm::omp::target::Int64Envar NumMPIComms("OMPTARGET_NUM_MPI_COMMS", 1);
// Maximum buffer Size to use during data transfer.
static llvm::omp::target::Int64Envar
    MPIFragmentSize("OMPTARGET_MPI_FRAGMENT_SIZE", 10);

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

  assert(false && "Every enum value must be checked on the switch above.");
  return nullptr;
}

// Coroutine events implementation
// =============================================================================
void EventTy::resume() {
  // Acquire first handle not done.
  const CoHandleTy &rootHandle = Handle.promise().rootHandle;
  auto &resumableHandle = rootHandle.promise().prevHandle;
  while (resumableHandle.done()) {
    resumableHandle = resumableHandle.promise().prevHandle;

    if (resumableHandle == rootHandle)
      break;
  }

  if (!resumableHandle.done())
    resumableHandle.resume();
}

void EventTy::wait() {
  // Advance the event progress until it is completed.
  while (!done()) {
    resume();

    std::this_thread::sleep_for(
        std::chrono::microseconds(EventPollingRate.get()));
  }
}

bool EventTy::done() const { return Handle.done(); }

bool EventTy::empty() const { return !Handle; }

llvm::Error &EventTy::getError() const {
  return Handle.promise().CoroutineError;
}

EventTy::~EventTy() {
  if (Handle)
    Handle.destroy();
}

// Helpers
// =============================================================================
MPIRequestManagerTy::~MPIRequestManagerTy() {
  assert(Requests.empty() && "Requests must be fulfilled and emptied before "
                             "destruction. Did you co_await on it?");
}

void MPIRequestManagerTy::send(const void *Buffer, int Size,
                               MPI_Datatype Datatype) {
  MPI_Isend(Buffer, Size, Datatype, OtherRank, Tag, Comm,
            &Requests.emplace_back(MPI_REQUEST_NULL));
}

void MPIRequestManagerTy::receive(void *Buffer, int Size,
                                  MPI_Datatype Datatype) {
  MPI_Irecv(Buffer, Size, Datatype, OtherRank, Tag, Comm,
            &Requests.emplace_back(MPI_REQUEST_NULL));
}

EventTy MPIRequestManagerTy::wait() {
  int RequestsCompleted = false;

  while (!RequestsCompleted) {
    int MPIError = MPI_Testall(Requests.size(), Requests.data(),
                               &RequestsCompleted, MPI_STATUSES_IGNORE);

    if (MPIError != MPI_SUCCESS)
      co_return createError("Waiting of MPI requests failed with code %d",
                            MPIError);

    co_await std::suspend_always{};
  }

  Requests.clear();

  co_return llvm::Error::success();
}

EventTy operator co_await(MPIRequestManagerTy &RequestManager) {
  return RequestManager.wait();
}

// Event Implementations
// =============================================================================

namespace OriginEvents {

EventTy allocateBuffer(MPIRequestManagerTy RequestManager, int64_t Size,
                       void **Buffer) {
  RequestManager.send(&Size, 1, MPI_INT64_T);

  RequestManager.receive(Buffer, sizeof(void *), MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy deleteBuffer(MPIRequestManagerTy RequestManager, void *Buffer) {
  RequestManager.send(&Buffer, sizeof(void *), MPI_BYTE);

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy submit(MPIRequestManagerTy RequestManager, void *OrgBuffer,
               void *DstBuffer, int64_t Size) {
  RequestManager.send(&DstBuffer, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);

  // Operates over many fragments of the original buffer of at most
  // MPI_FRAGMENT_SIZE bytes.
  char *BufferByteArray = reinterpret_cast<char *>(OrgBuffer);
  int64_t RemainingBytes = Size;
  while (RemainingBytes > 0) {
    RequestManager.send(
        &BufferByteArray[Size - RemainingBytes],
        static_cast<int>(std::min(RemainingBytes, MPIFragmentSize.get())),
        MPI_BYTE);
    RemainingBytes -= MPIFragmentSize.get();
  }

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy retrieve(MPIRequestManagerTy RequestManager, void *OrgBuffer,
                 void *DstBuffer, int64_t Size) {
  RequestManager.send(&DstBuffer, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);

  // Operates over many fragments of the original buffer of at most
  // MPI_FRAGMENT_SIZE bytes.
  char *BufferByteArray = reinterpret_cast<char *>(OrgBuffer);
  int64_t RemainingBytes = Size;
  while (RemainingBytes > 0) {
    RequestManager.receive(
        &BufferByteArray[Size - RemainingBytes],
        static_cast<int>(std::min(RemainingBytes, MPIFragmentSize.get())),
        MPI_BYTE);
    RemainingBytes -= MPIFragmentSize.get();
  }

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy exchange(MPIRequestManagerTy RequestManager, void *OrgBuffer,
                 int DstRank, void *DstBuffer, int64_t Size) {
  RequestManager.send(&OrgBuffer, sizeof(void *), MPI_BYTE);
  RequestManager.send(&DstRank, 1, MPI_INT);
  RequestManager.send(&DstBuffer, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy execute(MPIRequestManagerTy RequestManager,
                llvm::SmallVector<void *> Args, uint32_t TargetEntryIdx) {
  const size_t NumArgs = Args.size();
  RequestManager.send(&NumArgs, 1, MPI_UINT64_T);
  RequestManager.send(Args.data(), NumArgs * sizeof(void *), MPI_BYTE);
  RequestManager.send(&TargetEntryIdx, 1, MPI_INT32_T);

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy sync(EventPtr Event) {
  while (Event->done())
    co_await std::suspend_always{};

  co_return llvm::Error::success();
}

EventTy exit(MPIRequestManagerTy RequestManager) {
  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

} // namespace OriginEvents

namespace DestinationEvents {

EventTy allocateBuffer(MPIRequestManagerTy RequestManager) {
  int64_t Size = 0;
  RequestManager.receive(&Size, 1, MPI_INT64_T);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  void *Buffer = malloc(Size);
  RequestManager.send(&Buffer, sizeof(void *), MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy deleteBuffer(MPIRequestManagerTy RequestManager) {
  void *Buffer = nullptr;
  RequestManager.receive(&Buffer, sizeof(void *), MPI_BYTE);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  free(Buffer);

  // Event completion notification
  RequestManager.send(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy submit(MPIRequestManagerTy RequestManager) {
  void *Buffer = nullptr;
  int64_t Size = 0;
  RequestManager.receive(&Buffer, sizeof(void *), MPI_BYTE);
  RequestManager.receive(&Size, 1, MPI_INT64_T);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  // Operates over many fragments of the original buffer of at most
  // MPI_FRAGMENT_SIZE bytes.
  char *BufferByteArray = reinterpret_cast<char *>(Buffer);
  int64_t RemainingBytes = Size;
  while (RemainingBytes > 0) {
    RequestManager.receive(
        &BufferByteArray[Size - RemainingBytes],
        static_cast<int>(std::min(RemainingBytes, MPIFragmentSize.get())),
        MPI_BYTE);
    RemainingBytes -= MPIFragmentSize.get();
  }

  // Event completion notification
  RequestManager.send(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy retrieve(MPIRequestManagerTy RequestManager) {
  void *Buffer = nullptr;
  int64_t Size = 0;
  RequestManager.receive(&Buffer, sizeof(void *), MPI_BYTE);
  RequestManager.receive(&Size, 1, MPI_INT64_T);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  // Operates over many fragments of the original buffer of at most
  // MPI_FRAGMENT_SIZE bytes.
  char *BufferByteArray = reinterpret_cast<char *>(Buffer);
  int64_t RemainingBytes = Size;
  while (RemainingBytes > 0) {
    RequestManager.send(
        &BufferByteArray[Size - RemainingBytes],
        static_cast<int>(std::min(RemainingBytes, MPIFragmentSize.get())),
        MPI_BYTE);
    RemainingBytes -= MPIFragmentSize.get();
  }

  // Event completion notification
  RequestManager.send(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy exchange(MPIRequestManagerTy RequestManager,
                 EventSystemTy &EventSystem) {
  int DstRank = 0;
  void *OrgBuffer = nullptr;
  void *DstBuffer = nullptr;
  int64_t Size = 0;
  RequestManager.receive(&OrgBuffer, sizeof(void *), MPI_BYTE);
  RequestManager.receive(&DstRank, 1, MPI_INT);
  RequestManager.receive(&DstBuffer, sizeof(void *), MPI_BYTE);
  RequestManager.receive(&Size, 1, MPI_INT64_T);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  auto RemoteSubmit = EventSystem.createEvent(OriginEvents::submit, DstRank,
                                              OrgBuffer, DstBuffer, Size);
  if (auto Error = co_await *RemoteSubmit; Error)
    co_return Error;

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy execute(MPIRequestManagerTy RequestManager,
                __tgt_target_table *TargetTable) {

  size_t NumArgs = 0;
  RequestManager.receive(&NumArgs, 1, MPI_UINT64_T);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  llvm::SmallVector<void *> Args(NumArgs);
  RequestManager.receive(Args.data(), NumArgs * sizeof(uintptr_t), MPI_BYTE);
  uint32_t TargetEntryIdx = -1;
  RequestManager.receive(&TargetEntryIdx, sizeof(uint32_t), MPI_BYTE);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  // Iterates over all the host table entries to see if we can locate the
  // host_ptr.
  __tgt_offload_entry *Begin = TargetTable->EntriesBegin;
  __tgt_offload_entry *End = TargetTable->EntriesEnd;
  __tgt_offload_entry *Curr = Begin;

  // Iterates over all the table entries to see if we can locate the entry.
  void (*TargetEntry)(void) = nullptr;
  for (uint32_t I = 0; Curr < End; ++Curr, ++I) {
    if (I == TargetEntryIdx) {
      // We got a match, now fill the HostPtrToTableMap so that we may avoid
      // this search next time.
      *((void **)&TargetEntry) = Curr->addr;
      break;
    }
  }

  ffi_cif Cif{};
  llvm::SmallVector<ffi_type *> ArgsTypes(NumArgs);
  ffi_prep_cif(&Cif, FFI_DEFAULT_ABI, NumArgs, &ffi_type_void, &ArgsTypes[0]);
  ffi_call(&Cif, TargetEntry, NULL, &Args[0]);

  // Event completion notification
  RequestManager.send(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy exit(MPIRequestManagerTy RequestManager,
             std::atomic<EventSystemStateTy> &EventSystemState) {
  EventSystemStateTy OldState =
      EventSystemState.exchange(EventSystemStateTy::EXITED);
  assert(OldState != EventSystemStateTy::EXITED &&
         "Exit event received multiple times");

  // Event completion notification
  RequestManager.send(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

} // namespace DestinationEvents

// Event Queue implementation
// =============================================================================
EventQueue::EventQueue() : Queue(), QueueMtx(), CanPopCv() {}

size_t EventQueue::size() {
  std::lock_guard<std::mutex> lock(QueueMtx);
  return Queue.size();
}

void EventQueue::push(EventTy &&Event) {
  {
    std::unique_lock<std::mutex> lock(QueueMtx);
    Queue.emplace(Event);
  }

  // Notifies a thread possibly blocked by an empty queue.
  CanPopCv.notify_one();
}

EventTy EventQueue::pop() {
  std::unique_lock<std::mutex> lock(QueueMtx);

  // Waits for at least one item to be pushed.
  while (Queue.empty()) {
    const bool has_new_event = CanPopCv.wait_for(
        lock, std::chrono::microseconds(EventPollingRate.get()),
        [&] { return !Queue.empty(); });

    if (!has_new_event) {
      return {};
    }
  }

  assert(!Queue.empty() && "Queue was empty on pop operation.");

  EventTy TargetEvent = Queue.front();
  Queue.pop();
  return TargetEvent;
}

// Event System implementation
// =============================================================================
EventSystemTy::EventSystemTy()
    : EventSystemState(EventSystemStateTy::CREATED) {}

EventSystemTy::~EventSystemTy() {
  if (EventSystemState == EventSystemStateTy::FINALIZED)
    return;

  REPORT("Destructing internal event system before deinitializing it.\n");
  deinitialize();
}

bool EventSystemTy::initialize() {
  if (EventSystemState >= EventSystemStateTy::INITIALIZED) {
    REPORT("Trying to initialize event system twice.\n");
    return false;
  }

  if (!createLocalMPIContext())
    return false;

  EventSystemState = EventSystemStateTy::INITIALIZED;

  return true;
}

bool EventSystemTy::deinitialize() {
  if (EventSystemState == EventSystemStateTy::FINALIZED) {
    REPORT("Trying to deinitialize event system twice.\n");
    return false;
  }

  if (EventSystemState == EventSystemStateTy::RUNNING) {
    REPORT("Trying to deinitialize event system while it is running.\n");
    return false;
  }

  // Only send exit events from the host side
  if (isHead() && WorldSize > 1) {
    const int NumWorkers = WorldSize - 1;
    llvm::SmallVector<EventPtr> ExitEvents(NumWorkers);
    for (int WorkerRank = 0; WorkerRank < NumWorkers; WorkerRank++) {
      ExitEvents[WorkerRank] = createEvent(OriginEvents::exit, WorkerRank);
      ExitEvents[WorkerRank]->resume();
    }

    bool SuccessfullyExited = true;
    for (int WorkerRank = 0; WorkerRank < NumWorkers; WorkerRank++) {
      ExitEvents[WorkerRank]->wait();
      SuccessfullyExited &= ExitEvents[WorkerRank]->done();
    }

    if (!SuccessfullyExited) {
      REPORT("Failed to stop worker processes.\n");
      return false;
    }
  }

  if (!destroyLocalMPIContext())
    return false;

  EventSystemState = EventSystemStateTy::FINALIZED;

  return true;
}

void EventSystemTy::runEventHandler(EventQueue &Queue) {
  while (EventSystemState == EventSystemStateTy::RUNNING || Queue.size() > 0) {
    EventTy Event = Queue.pop();

    // Re-checks the stop condition when no event was found.
    if (Event.empty()) {
      continue;
    }

    Event.resume();

    if (!Event.done()) {
      Queue.push(std::move(Event));
    }

    if (Event.getError())
      REPORT("Internal event failed with msg: %s\n",
             toString(std::move(Event.getError())).data());
  }
}

void EventSystemTy::runGateThread(__tgt_target_table *TargetTable) {
  // Updates the event state and
  EventSystemState = EventSystemStateTy::RUNNING;

  // Spawns the event handlers.
  llvm::SmallVector<std::thread> EventHandlers;
  EventHandlers.resize(NumExecEventHandlers.get() + NumDataEventHandlers.get());
  for (int Idx = 0; Idx < EventHandlers.size(); Idx++) {
    EventHandlers[Idx] = std::thread(&EventSystemTy::runEventHandler, this,
                                     std::ref(Idx < NumExecEventHandlers.get()
                                                  ? ExecEventQueue
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
          std::chrono::microseconds(EventPollingRate.get()));
      continue;
    }

    // Acquires the event information from the received request, which are:
    // - Event type
    // - Event tag
    // - Target comm
    // - Event source rank
    int EventInfo[2];
    MPI_Mrecv(EventInfo, 2, MPI_INT, &EventReqMsg, &EventStatus);
    const auto NewEventType = static_cast<EventTypeTy>(EventInfo[0]);
    MPIRequestManagerTy RequestManager(getNewEventComm(EventInfo[1]),
                                       EventInfo[1], EventStatus.MPI_SOURCE);

    // Creates a new receive event of 'event_type' type.
    using namespace DestinationEvents;
    EventTy NewEvent;
    switch (NewEventType) {
    case EventTypeTy::ALLOC:
      NewEvent = allocateBuffer(std::move(RequestManager));
      break;
    case EventTypeTy::DELETE:
      NewEvent = deleteBuffer(std::move(RequestManager));
      break;
    case EventTypeTy::SUBMIT:
      NewEvent = submit(std::move(RequestManager));
      break;
    case EventTypeTy::RETRIEVE:
      NewEvent = retrieve(std::move(RequestManager));
      break;
    case EventTypeTy::EXCHANGE:
      NewEvent = exchange(std::move(RequestManager), *this);
      break;
    case EventTypeTy::EXECUTE:
      NewEvent = execute(std::move(RequestManager), TargetTable);
      break;
    case EventTypeTy::EXIT:
      NewEvent = exit(std::move(RequestManager), EventSystemState);
      break;
    case EventTypeTy::SYNC:
      assert(false && "Trying to create a local event on a remote node");
    }

    if (NewEventType == EventTypeTy::EXECUTE) {
      ExecEventQueue.push(std::move(NewEvent));
    } else {
      DataEventQueue.push(std::move(NewEvent));
    }
  }

  assert(EventSystemState == EventSystemStateTy::EXITED &&
         "Event State should be EXITED after receiving an Exit event");

  // Waits for the Event Handler threads.
  for (auto &EventHandler : EventHandlers) {
    if (EventHandler.joinable()) {
      EventHandler.join();
    } else {
      assert(false && "Event Handler threads not joinable at the end of gate "
                      "thread logic.");
    }
  }
}

// Creates a new event tag of at least 'FIRST_EVENT' value.
// Tag values smaller than 'FIRST_EVENT' are reserved for control
// messages between the event systems of different MPI processes.
int EventSystemTy::createNewEventTag() {
  int tag = 0;

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
  EventCommPool.resize(NumMPIComms.get(), MPI_COMM_NULL);
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
