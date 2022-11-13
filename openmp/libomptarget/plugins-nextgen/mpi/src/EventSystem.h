//===------- event_system.h - Concurrent MPI communication ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the MPI Event System used by the MPI
// target.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_OMPCLUSTER_EVENT_SYSTEM_H_
#define _OMPTARGET_OMPCLUSTER_EVENT_SYSTEM_H_

#include <atomic>
#include <cassert>
#include <concepts>
#include <condition_variable>
#include <coroutine>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>

#define MPICH_SKIP_MPICXX
#include <mpi.h>

#include "llvm/ADT/SmallVector.h"

#include "Utilities.h"

// External forward declarations.
// =============================================================================
struct __tgt_target_table;

// Internal forward declarations and type aliases.
// =============================================================================
struct EventTy;

/// Automaticaly managed event pointer.
///
/// \note: Every event must always be accessed/stored in a shared_ptr structure.
/// This allows for automatic memory management among the many threads of the
/// libomptarget runtime.
using EventPtr = std::shared_ptr<EventTy>;

// Helper
// =============================================================================
template <typename... ArgsTy>
static llvm::Error createError(const char *ErrFmt, ArgsTy... Args) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), ErrFmt,
                                 Args...);
}

/// The event type (type of action it will performed).
///
/// Enumerates the available events. Each enum item should be accompanied by an
/// event class derived from BaseEvent. All the events are executed at a remote
/// MPI process target by the event.
enum class EventTypeTy : int {
  // Memory management.
  ALLOC,  // Allocates a buffer at the remote process.
  DELETE, // Deletes a buffer at the remote process.

  // Data movement.
  SUBMIT,   // Sends a buffer data to a remote process.
  RETRIEVE, // Receives a buffer data from a remote process.
  EXCHANGE, // Exchange a buffer between two remote processes.

  // Target region execution.
  EXECUTE, // Executes a target region at the remote process.

  // Local event used to wait on other events.
  SYNC,

  // Internal event system commands.
  EXIT // Stops the event system execution at the remote process.
};

/// EventType to string conversion.
///
/// \returns the string representation of \p type.
const char *toString(EventTypeTy type);

// Coroutine events
// =============================================================================
// Return object for the event system coroutines. This class works as an
// external handle for the coroutine execution, allowing anyone to: query for
// the coroutine completion, resume the coroutine and check its state. Moreover,
// this class allows for coroutines to be chainable, meaning a coroutine
// function may wait on the completion of another one by using the co_await
// operator, all through a single external handle.
struct EventTy {
  // Internal event handle to access C++ coroutine states.
  struct promise_type;
  using CoHandleTy = std::coroutine_handle<promise_type>;
  CoHandleTy Handle;

  // Internal (and required) promise type. Allows for customization of the
  // coroutines behavior and to store custom data inside the coroutine itself.
  struct promise_type {
    // Coroutines are chained as a reverse linked-list. The most-recent
    // coroutine in a chain points to the previous one and so on, until the root
    // (and first) coroutine, which then points to the most-recent one. The root
    // always refers to the coroutine stored in the external handle, the only
    // handle an external user have access to.
    CoHandleTy prevHandle;
    CoHandleTy rootHandle;
    // Indicates if the coroutine was completed successfully. Contains the
    // appropriate error otherwise.
    llvm::Error CoroutineError;

    promise_type() : CoroutineError(llvm::Error::success()) {
      prevHandle = rootHandle = CoHandleTy::from_promise(*this);
    }

    // Event coroutines should always suspend upon creation and finalization.
    std::suspend_always initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }

    // Coroutines should return llvm::Error::success() or an appropriate error
    // message.
    void return_value(llvm::Error &&GivenError) noexcept {
      CoroutineError = std::move(GivenError);
    }

    // Any unhandled exception should create an externally visible error.
    void unhandled_exception() {
      assert(std::uncaught_exceptions() > 0 &&
             "Function should only be called if an uncaught exception is "
             "generated inside the coroutine");
      CoroutineError = createError("Event generated an unhandled exception");
    }

    // Returns the external coroutine handle from the promise object.
    EventTy get_return_object() {
      return {.Handle = CoHandleTy::from_promise(*this)};
    }
  };

  // Automatically destroys valid coroutine handles.
  ~EventTy();

  // Execution handling.
  // Resume the coroutine execution up until the next suspension point.
  void resume();
  // Blocks the caller thread until the coroutine is completed.
  void wait();
  // Checks if the coroutine is completed or not.
  bool done() const;

  // Coroutine state handling.
  // Checks if the coroutine is valid.
  bool empty() const;
  // Get the returned error from the coroutine.
  llvm::Error &getError() const;

  // EventTy instances are also awaitables. This means one can link multiple
  // EventTy together by calling the co_await operator on one another. For this
  // to work, EventTy must implement the following three functions.

  // Called on the new coroutine before suspending the current one on co_await.
  // If returns true, the new coroutine is already completed, thus it should not
  // be linked against the current one and the current coroutine can continue
  // without suspending.
  bool await_ready() { return Handle.done(); }
  // Called on the new coroutine when the current one is suspended. It is
  // responsible for chaining coroutines together.
  void await_suspend(CoHandleTy suspendedHandle) {
    auto &currPromise = Handle.promise();
    auto &suspendedPromise = suspendedHandle.promise();
    auto &rootPromise = suspendedPromise.rootHandle.promise();

    currPromise.prevHandle = suspendedHandle;
    currPromise.rootHandle = suspendedPromise.rootHandle;

    rootPromise.prevHandle = Handle;
  }
  // Called on the new coroutine when the current one is resumed. Used to return
  // errors when co_awaiting on other EventTy.
  llvm::Error await_resume() {
    return std::move(Handle.promise().CoroutineError);
  }
};

// Coroutine like manager for many non-blocking MPI calls. Allows for coroutine
// to co_await on the registered MPI requests.
class MPIRequestManagerTy {
  // Target specification for the MPI messages.
  const MPI_Comm Comm;
  const int Tag;
  const int OtherRank;
  // Pending MPI requests.
  llvm::SmallVector<MPI_Request> Requests;

public:
  MPIRequestManagerTy(MPI_Comm Comm, int Tag, int OtherRank,
                      llvm::SmallVector<MPI_Request> InitialRequests = {})
      : Comm(Comm), Tag(Tag), OtherRank(OtherRank), Requests(InitialRequests) {}

  ~MPIRequestManagerTy();

  // Sends a buffer of given datatype items with determined size to target.
  void send(const void *Buffer, int Size, MPI_Datatype Datatype);

  // Receives a buffer of given datatype items with determined size from target.
  void receive(void *Buffer, int Size, MPI_Datatype Datatype);

  // Coroutine that waits on all internal pending requests.
  EventTy wait();
};

// Coroutine events created at the origin rank of the event.
namespace OriginEvents {

EventTy allocateBuffer(MPIRequestManagerTy RequestManager, int64_t Size,
                       void **Buffer);
EventTy deleteBuffer(MPIRequestManagerTy RequestManager, void *Buffer);
EventTy submit(MPIRequestManagerTy RequestManager, void *OrgBuffer,
               void *DstBuffer, int64_t Size);
EventTy retrieve(MPIRequestManagerTy RequestManager, void *OrgBuffer,
                 void *DstBuffer, int64_t Size);
EventTy exchange(MPIRequestManagerTy RequestManager, void *OrgBuffer,
                 int DstRank, void *DstBuffer, int64_t Size);
EventTy execute(MPIRequestManagerTy RequestManager,
                llvm::SmallVector<void *> Args, uint32_t TargetEntryIdx);
EventTy sync(EventPtr Event);
EventTy exit(MPIRequestManagerTy RequestManager);

// Transform a function pointer to its representing enumerator.
template <typename FuncTy> constexpr EventTypeTy toEventType(FuncTy) {
  if constexpr (std::is_same_v<FuncTy, decltype(allocateBuffer)>)
    return EventTypeTy::ALLOC;
  else if constexpr (std::is_same_v<FuncTy, decltype(deleteBuffer)>)
    return EventTypeTy::DELETE;
  else if constexpr (std::is_same_v<FuncTy, decltype(submit)>)
    return EventTypeTy::SUBMIT;
  else if constexpr (std::is_same_v<FuncTy, decltype(retrieve)>)
    return EventTypeTy::RETRIEVE;
  else if constexpr (std::is_same_v<FuncTy, decltype(exchange)>)
    return EventTypeTy::EXCHANGE;
  else if constexpr (std::is_same_v<FuncTy, decltype(execute)>)
    return EventTypeTy::EXECUTE;
  else if constexpr (std::is_same_v<FuncTy, decltype(sync)>)
    return EventTypeTy::SYNC;
  else if constexpr (std::is_same_v<FuncTy, decltype(exit)>)
    return EventTypeTy::EXIT;

  assert(false && "Invalid event function");
}

} // namespace OriginEvents

// Event Queue
// =============================================================================
/// Event queue for received events.
class EventQueue {
private:
  /// Base internal queue.
  std::queue<EventTy> Queue;
  /// Base queue sync mutex.
  std::mutex QueueMtx;

  /// Conditional variables to block popping on an empty queue.
  std::condition_variable CanPopCv;

public:
  /// Event Queue default constructor.
  EventQueue();

  /// Gets current queue size.
  size_t size();

  /// Push an event to the queue, resizing it when needed.
  void push(EventTy &&Event);

  /// Pops an event from the queue, returning nullptr if the queue is empty.
  EventTy pop();
};

// Event System
// =============================================================================

/// MPI tags used in control messages.
///
/// Special tags values used to send control messages between event systems of
/// different processes. When adding new tags, please summarize the tag usage
/// with a side comment as done below.
enum class ControlTagsTy : int {
  EVENT_REQUEST = 0, // Used by event handlers to receive new event requests.
  FIRST_EVENT        // Tag used by the first event. Must always be placed last.
};

/// Event system execution state.
///
/// Describes the event system state through the program.
enum class EventSystemStateTy {
  CREATED,     // ES was created but it is not ready to send or receive new
               // events.
  INITIALIZED, // ES was initialized alongside internal MPI states. It is ready
               // to send new events, but not receive them.
  RUNNING,     // ES is running and ready to receive new events.
  EXITED,      // ES was stopped.
  FINALIZED    // ES was finalized and cannot run anything else.
};

/// The distributed event system.
class EventSystemTy {
  // MPI definitions.
  /// The largest MPI tag allowed by its implementation.
  int32_t MPITagMaxValue = 0;
  /// Communicator used by the gate thread and base communicator for the event
  /// system.
  MPI_Comm GateThreadComm = MPI_COMM_NULL;
  /// Communicator pool distributed over the events. Many MPI implementations
  /// allow for better network hardware parallelism when unrelated MPI messages
  /// are exchanged over distinct communicators. Thus this pool will be given in
  /// a round-robin fashion to each newly created event to better utilize the
  /// hardware capabilities.
  llvm::SmallVector<MPI_Comm> EventCommPool{};
  /// Number of process used by the event system.
  int WorldSize = -1;
  /// The local rank of the current instance.
  int LocalRank = -1;

  /// Number of events created by the current instance so far. This is used to
  /// generate unique MPI tags for each event.
  std::atomic<int> EventCounter{0};

  /// Event queue between the local gate thread and the event handlers. The exec
  /// queue is responsible for only running the execution events, while the data
  /// queue executes all the other ones. This allows for long running execution
  /// events to not block any data transfers (which are all done in a
  /// non-blocking fashion).
  EventQueue ExecEventQueue{};
  EventQueue DataEventQueue{};

  /// Event System execution state.
  std::atomic<EventSystemStateTy> EventSystemState{};

private:
  /// Function executed by the event handler threads.
  void runEventHandler(EventQueue &Queue);

  /// Creates a new unique event tag for a new event.
  int createNewEventTag();

  /// Gets a comm for a new event from the comm pool.
  MPI_Comm &getNewEventComm(int MPITag);

public:
  /// Creates a local MPI context containing a exclusive comm for the gate
  /// thread, and a comm pool to be used internally by the events. It also
  /// acquires the local MPI process description.
  bool createLocalMPIContext();

  /// Destroy the local MPI context and all of its comms.
  bool destroyLocalMPIContext();

  EventSystemTy();
  ~EventSystemTy();

  bool initialize();
  bool deinitialize();

  /// Creates a new event.
  ///
  /// Creates a new event of 'EventClass' type targeting the 'DestRank'. The
  /// 'args' parameters are additional arguments that may be passed to the
  /// EventClass origin constructor.
  ///
  /// /note: since this is a template function, it must be defined in
  /// this header.
  template <class EventFuncTy, typename... ArgsTy>
  requires std::invocable<EventFuncTy, MPIRequestManagerTy, ArgsTy...> EventPtr
  createEvent(EventFuncTy EventFunc, int DstDeviceID, ArgsTy &&...Args);

  /// Gate thread procedure.
  ///
  /// Caller thread will spawn the event handlers, execute the gate logic and
  /// wait until the event system receive an Exit event.
  void runGateThread(__tgt_target_table *TargetTable);

  /// Get the number of workers available.
  ///
  /// \return the number of MPI available workers.
  int getNumWorkers() const;

  /// Check if we are at the host MPI process.
  ///
  /// \return true if the current MPI process is the host (rank 0), false
  /// otherwise.
  int isHead() const;
};

template <class EventFuncTy, typename... ArgsTy>
requires std::invocable<EventFuncTy, MPIRequestManagerTy, ArgsTy...>
    EventPtr EventSystemTy::createEvent(EventFuncTy EventFunc, int DstDeviceID,
                                        ArgsTy &&...Args) {
  auto NotificationSender = [this](EventFuncTy EventFunc, int DstDeviceID,
                                   ArgsTy &&...Args) -> EventTy {
    // Create event MPI request manager.
    // MPI rank 0 is our head node/host. Worker rank starts at 1.
    const int DstDeviceRank = DstDeviceID + 1;
    const int EventTag = createNewEventTag();
    auto &EventComm = getNewEventComm(EventTag);

    // Send new event notification.
    int EventNotificationInfo[] = {
        static_cast<int>(OriginEvents::toEventType(EventFunc)), EventTag};
    MPI_Request NotificationRequest = MPI_REQUEST_NULL;
    int MPIError = MPI_Isend(EventNotificationInfo, 2, MPI_INT, DstDeviceID,
                             static_cast<int>(ControlTagsTy::EVENT_REQUEST),
                             GateThreadComm, &NotificationRequest);

    if (MPIError != MPI_SUCCESS)
      co_return createError(
          "MPI failed during event notification with error %d", MPIError);

    MPIRequestManagerTy RequestManager(EventComm, EventTag, DstDeviceRank,
                                       {NotificationRequest});

    co_return (co_await EventFunc(std::move(RequestManager), Args...));
  };

  return std::make_shared<EventTy>(std::move(NotificationSender(
      EventFunc, DstDeviceID, std::forward<ArgsTy>(Args)...)));
}

#endif // _OMPTARGET_OMPCLUSTER_EVENT_SYSTEM_H_
