//===------- event_system.h - Concurrent MPI communicaiton ------*- C++ -*-===//
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
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>

#define MPICH_SKIP_MPICXX
#include <mpi.h>

#include "llvm/ADT/SmallVector.h"

// External forward declarations.
// =============================================================================
class MPIManagerTy;
struct __tgt_target_table;

// Internal forward declarations and type aliases.
// =============================================================================
class BaseEventTy;
enum class EventSystemStateTy;

/// Automaticaly managed event pointer.
///
/// \note: Every event must always be accessed/stored in a shared_ptr structure.
/// This allows for automatic memory management among the many threads of the
/// libomptarget runtime.
using EventPtr = std::shared_ptr<BaseEventTy>;

// Event Types
// =============================================================================
/// The event location.
///
/// Enumerates whether an event is executing at its two possible locations:
/// its origin or its destination.
enum class EventLocationTy : bool { ORIG = 0, DEST };

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
  RETRIEVE, // Receives a buffer data from a remote process.
  SUBMIT,   // Sends a buffer data to a remote process.
  EXCHANGE, // Exchange a buffer between two remote processes.

  // Target region execution.
  EXECUTE, // Executes a target region at the remote process.

  // Local event used to wait on other events.
  SYNC,

  // Internal event system commands.
  EXIT // Stops the event system execution at the remote process.
};

/// The event execution state.
///
/// Enumerates the event states during its lifecycle through the event system.
/// New states should be added to the enum in the order they happen at the event
/// system.
enum class EventStateTy : int {
  CREATED = 0, // Event was only create but it was not executed yet.
  EXECUTING,   // Event is currently being executed in the background and it
               // is registering new MPI requests.
  WAITING,     // Event was executed and is now waiting on the MPI requests
               // complete.
  FAILED,      // Event failed during execution
  FINISHED     // The event and its MPI requests are completed.
};

/// EventLocation to string conversion.
///
/// \returns the string representation of \p location.
const char *toString(EventLocationTy location);

/// EventType to string conversion.
///
/// \returns the string representation of \p type.
const char *toString(EventTypeTy type);

/// EventState to string conversion.
///
/// \returns the string representation of \p state.
const char *toString(EventStateTy state);

// Events
// =============================================================================

/// The base event of all event types.
///
/// This class contains both the common data stored and common procedures
/// executed at all events. New events that derive from this class must comply
/// with the following:
/// - Declare the new EventType item;
/// - Name the derived class as the concatenation of the new EventType name and
///   the "Event" word. E.g.: ALLOC event -> class AllocEvent;
/// - Implement both pure virtual functions:
///    - #runOrigin;
///    - #runDestination;
/// - Implement two constructors (one for each EventLocation) with this
///   prototype:
///
///   Event(int MPITag, MPI_Comm TargetComm, int OrigRank, int
///   DestRank,...);
///
class BaseEventTy {
public:
  /// The only (non-child) class that can access the Event protected members.
  friend class EventSystemTy;

  // Event definitions.
  /// Location that the event is being executed: origin or destination ranks.
  const EventLocationTy EventLocation;
  /// Event type that represents its actions.
  const EventTypeTy EventType;

  // MPI definitions.
  /// MPI communicator to be used by the event.
  const MPI_Comm TargetComm;
  /// MPI tag that must be used on every MPI communication of the event.
  const int MPITag;
  /// Rank of the process that created the event.
  const int OrigRank;
  /// Rank of the process that was target by the event.
  const int DestRank;

protected:
  using LabelPointer = void *;
  /// Event coroutine state.
  LabelPointer ResumeLocation = nullptr;

private:
  /// MPI non-blocking requests to be synchronized during event execution.
  llvm::SmallVector<MPI_Request> PendingRequests;

  /// The event execution state.
  std::atomic<EventStateTy> EventState{EventStateTy::CREATED};

  /// Call-guard for the #progress function.
  ///
  /// This atomic ensures only one thread executes the #progress code at a time.
  std::atomic<bool> ProgressGuard{false};

  /// Parameters used to create the event at the destination process.
  ///
  /// \note this array is needed so non-blocking MPI messages can be used to
  /// create the event.
  uint32_t InitialRequestInfo[2] = {0, 0};

public:
  BaseEventTy &operator=(BaseEventTy &) = delete;
  BaseEventTy(BaseEventTy &) = delete;

  /// Advance the progress of the event.
  ///
  /// \note calling this function once does n;;;;;ot guarantee that the event is
  /// completely executed. One must call this function until #isDone returns
  /// true.
  void progress();

  /// Check if the event is completed.
  ///
  /// \return true if the event is completed.
  bool isDone() const;

  /// Wait for the event to be completed.
  ///
  /// Waits for the completion of the event on both the local and remote
  /// processes. This function will choose between the blocking and tasking
  /// implementations depending if the current code is inside a task or not.
  ///
  /// \note this function is almost equivalent to calling #runStage while
  /// #isDone is returning false.
  void wait();

  /// Get the current event execution state.
  ///
  /// \returns The current EventState.
  EventStateTy getEventState() const;

protected:
  /// BaseEvent constructor.
  BaseEventTy(EventLocationTy EventLocation, EventTypeTy EventType, int MPITag,
              MPI_Comm TargetComm, int OrigRank, int DestRank);

  /// BaseEvent default destructor.
  virtual ~BaseEventTy() = default;

  /// Push a new MPI request to the request array.
  ///
  /// \returns a reference to the next available MPI request at
  /// #pending_requests.
  MPI_Request *getNextRequest();

  /// Test if all pending requests have finished.
  ///
  /// \return true if all pending MPI requests are completed, false otherwise.
  bool checkPendingRequests();

private:
  /// Sends a new event notification to the destination.
  void notifyNewEvent();

  /// Calls #runOrigin or #runDestination coroutine.
  ///
  /// \returns true if the event run coroutine has finished, false otherwise.
  bool runCoroutine();

  // Event coroutines
  /// Executes the origin side of the event locally.
  virtual bool runOrigin() = 0;
  /// Executes the destination side of the event locally.
  virtual bool runDestination() = 0;
};

/// Allocates a buffer at a remote process.
class AllocEventTy final : public BaseEventTy {
private:
  /// Size of the buffer to be allocated.
  int64_t Size = 0;
  /// Pointer to variable to be filled with the address of the allocated buffer.
  void **AllocatedAddressPtr = nullptr;
  // Allocated address to be send back.
  void *DestAddress = 0;

public:
  /// Origin constructor.
  AllocEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank, int DestRank,
               int64_t Size, void **AllocatedAddress);

  /// Destination constructor.
  AllocEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank, int DestRank);

private:
  /// Sends the size and receives the allocated address.
  bool runOrigin() override;
  /// Receives the size, allocating the data and sending its address.
  bool runDestination() override;
};

/// Frees a buffer at a remote process.
class DeleteEventTy final : public BaseEventTy {
private:
  /// Address of the buffer to be freed.
  void *TargetAddress = 0;

public:
  /// Origin constructor.
  DeleteEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank, int DestRank,
                void *TargetAddress);

  /// Destination constructor.
  DeleteEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank, int DestRank);

private:
  /// Sends the address and waits for a notification of the data deletion.
  bool runOrigin() override;
  /// Receives the address, frees it and send a completion notification.
  bool runDestination() override;
};

/// Retrieves a buffer from a remote process.
class RetrieveEventTy final : public BaseEventTy {
private:
  /// Address of the origin's buffer to be filled with the destination's data.
  void *OrigPtr = nullptr;
  /// Address of the destination's buffer to be retrieved.
  const void *DestPtr = nullptr;
  /// Size of both the origin's and destination's buffers.
  int64_t Size = 0;

public:
  /// Origin constructor.
  RetrieveEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank, int DestRank,
                  void *OrigPtr, const void *DestPtr, int64_t Size);

  /// Destination constructor.
  RetrieveEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank, int DestRank);

private:
  /// Sends the buffer info and retries its data.
  bool runOrigin() override;
  /// Receives the buffer info and sends its data.
  bool runDestination() override;
};

/// Send a buffer to a remote process.
class SubmitEventTy final : public BaseEventTy {
private:
  /// Address of the origin's buffer to be submitted.
  const void *OrigPtr = nullptr;
  /// Address of the destination's buffer to be filled with the origin's data.
  void *DestPtr = nullptr;
  /// Size of both the origin's and destination's buffers.
  int64_t Size = 0;

public:
  /// Origin constructor.
  // TODO: Change to dest first then origin (target then host)
  SubmitEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank, int DestRank,
                const void *OrigPtr, void *DestPtr, int64_t Size);

  /// Destination constructor.
  SubmitEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank, int DestRank);

private:
  /// Sends the buffer info and then the buffer itself.
  bool runOrigin() override;
  /// Receives the buffer info and then the buffer itself.
  bool runDestination() override;

  friend class PackedSubmitEvent;
};

/// Exchange a buffer between two remote processes.
class ExchangeEventTy final : public BaseEventTy {
private:
  /// MPI rank of the data destination process.
  int DataDestRank = 0;
  /// Address of the data at the data source process.
  const void *SrcPtr = nullptr;
  /// Address of the data at the data destination process.
  void *DstPtr = nullptr;
  /// Size of both the data source's and data destination's buffers.
  int64_t Size = 0;
  /// Pointer to the remote submit event created at the remote source location.
  EventPtr RemoteSubmitEvent = nullptr;

public:
  /// Origin constructor.
  ExchangeEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank, int DestRank,
                  int DataDestRank, const void *SrcPtr, void *DstPtr,
                  int64_t Size);

  /// Destination constructor.
  ExchangeEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank, int DestRank);

private:
  /// Sends the buffer info.
  bool runOrigin() override;
  /// Receives the buffer info and start a SubmitEvent to the dest process.
  bool runDestination() override;
};

/// Executes a target region at a remote process.
class ExecuteEventTy final : public BaseEventTy {
private:
  /// Number of arguments of the target region.
  int32_t NumArgs = 0;
  /// Arguments of the target region.
  llvm::SmallVector<void *> Args{};
  /// Index of the target region.
  uint32_t TargetEntryIdx = -1;
  // Local target table with entry addresses.
  __tgt_target_table *TargetTable = nullptr;

public:
  /// Origin constructor.
  ExecuteEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank, int DestRank,
                 int32_t NumArgs, void **Args, uint32_t TargetEntryIdx);

  /// Destination constructor.
  ExecuteEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank, int DestRank,
                 __tgt_target_table *TargetTable);

private:
  /// Sends the target region info and wait for the completion notification.
  bool runOrigin() override;
  /// Receives the target region info, executes it and sends the notification.
  bool runDestination() override;
};

/// Local event used to wait on other events.
class SyncEventTy final : public BaseEventTy {
private:
  EventPtr TargetEvent;

public:
  /// Destination constructor.
  SyncEventTy(EventPtr &TargetEvent);

private:
  /// Does nothing.
  bool runOrigin() override;
  /// Waits for target_event to complete.
  bool runDestination() override;
};

/// Notify a remote process to stop its event system.
class ExitEventTy final : public BaseEventTy {
private:
  /// Pointer to the event system state.
  std::atomic<EventSystemStateTy> *EventSystemState = nullptr;

public:
  /// Origin constructor.
  ExitEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank, int DestRank);

  /// Destination constructor.
  ExitEventTy(int MPITag, MPI_Comm TargetComm, int OrigRank, int DestRank,
              std::atomic<EventSystemStateTy> *EventSystemState);

private:
  /// Just waits for the completion notification.
  bool runOrigin() override;
  /// Stops its event system and sends the notification.
  bool runDestination() override;
};

// Event Queue
// =============================================================================
/// Event queue for received events.
class EventQueue {
private:
  /// Base internal queue.
  std::queue<EventPtr> Queue;
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
  void push(EventPtr &Event);

  /// Pops an event from the queue, returning nullptr if the queue is empty.
  EventPtr pop();
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
  EXITED       // ES was stopped.
};

/// The distributed event system.
class EventSystemTy {
public:
  /// The largest MPI tag allowed by its implementation.
  static int32_t MPITagMaxValue;

  /// Communicator used by the gate thread.
  // TODO: Find a better way to share this with all the events. static is not
  // that great.
  static MPI_Comm GateThreadComm;

private:
  // MPI definitions.
  /// Communicator pool distributed over the events.
  llvm::SmallVector<MPI_Comm> EventCommPool{};
  /// Number of process used by the event system.
  int WorldSize = -1;
  /// The local rank of the current instance.
  int LocalRank = -1;

  /// Number of event created by the current instance.
  std::atomic<uint32_t> EventCounter{0};

  /// Event queue between the local gate thread and event handlers.
  EventQueue ExecEventQueue{};
  EventQueue DataEventQueue{};

  /// Event System execution state.
  std::atomic<EventSystemStateTy> EventSystemState{};

  bool IsInitialized = false;

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
  template <class EventClass, typename... ArgsTy>
  EventPtr createEvent(int DestRank, ArgsTy &&...Args);

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

template <class EventClassTy, typename... ArgsTy>
EventPtr EventSystemTy::createEvent(int DstDeviceID, ArgsTy &&...Args) {
  static_assert(std::is_convertible_v<EventClassTy *, BaseEventTy *>,
                "Cannot create an event from a class that is not derived from "
                "the BaseEvent class.");
  using MPITagTy = int;
  using RankTy = int;
  static_assert(std::is_constructible_v<EventClassTy, MPITagTy, MPI_Comm,
                                        RankTy, RankTy, ArgsTy...>,
                "Cannot create an event from the given argument types.");

  // MPI rank 0 is our head node/host. Worker rank starts at 1.
  const int DstDeviceRank = DstDeviceID + 1;

  const int EventTag = createNewEventTag();
  auto &EventComm = getNewEventComm(EventTag);

  EventPtr Event = std::make_shared<EventClassTy>(
      EventTag, EventComm, LocalRank, DstDeviceRank,
      std::forward<ArgsTy>(Args)...);

  return Event;
}

#endif // _OMPTARGET_OMPCLUSTER_EVENT_SYSTEM_H_
