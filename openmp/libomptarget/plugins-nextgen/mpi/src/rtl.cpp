//===------RTLs/mpi/src/rtl.cpp - Target RTLs Implementation - C++ ------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL NextGen for MPI applications
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <string>

#include "Debug.h"
#include "DeviceEnvironment.h"
#include "GlobalHandler.h"
#include "PluginInterface.h"
#include "Utilities.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DynamicLibrary.h"

#include "EventSystem.h"

namespace llvm::omp::target::plugin {

/// Forward declarations for all specialized data structures.
struct MPIPluginTy;
struct MPIDeviceTy;
struct MPIDeviceImageTy;
struct MPIEventQueueRef;
class MPIGlobalHandlerTy;
class MPIEventQueueManagerTy;

using llvm::sys::DynamicLibrary;

// TODO: Should this be defined inside the EventSystem?
using MPIEventQueue = SmallVector<EventPtr>;
using MPIEventQueuePtr = MPIEventQueue *;

/// Class implementing the MPI device images properties.
struct MPIDeviceImageTy : public DeviceImageTy {
  /// Create the MPI image with the id and the target image pointer.
  MPIDeviceImageTy(int32_t ImageId, const __tgt_device_image *TgtImage)
      : DeviceImageTy(ImageId, TgtImage), DynLib() {}

  /// Getter and setter for the dynamic library.
  DynamicLibrary &getDynamicLibrary() { return DynLib; }
  void setDynamicLibrary(const DynamicLibrary &Lib) { DynLib = Lib; }

private:
  /// The dynamic library that loaded the image.
  DynamicLibrary DynLib;
};

class MPIGlobalHandlerTy final : public GenericGlobalHandlerTy {
public:
  Error getGlobalMetadataFromDevice(GenericDeviceTy &GenericDevice,
                                    DeviceImageTy &Image,
                                    GlobalTy &DeviceGlobal) override {
    const char *GlobalName = DeviceGlobal.getName().data();
    MPIDeviceImageTy &MPIImage = static_cast<MPIDeviceImageTy &>(Image);

    // Get dynamic library that has loaded the device image.
    DynamicLibrary &DynLib = MPIImage.getDynamicLibrary();

    // Get the address of the symbol.
    void *Addr = DynLib.getAddressOfSymbol(GlobalName);
    if (Addr == nullptr) {
      return Plugin::error("Failed to load global '%s'", GlobalName);
    }

    // Save the pointer to the symbol.
    DeviceGlobal.setPtr(Addr);

    return Plugin::success();
  }
};

// MPI resource reference and queue
// =============================================================================
template <typename ResourceTy>
struct MPIResourceRef final : public GenericDeviceResourceRef {
  MPIResourceRef() {}

  MPIResourceRef(ResourceTy *Queue) : Resource(Queue) {}

  Error create() override {
    if (Resource)
      return Plugin::error("Recreating an existing resource");

    Resource = new ResourceTy;
    if (!Resource)
      return Plugin::error("Failed to allocated a new resource");

    return Plugin::success();
  }

  Error destroy() override {
    if (!Resource)
      return Plugin::error("Destroying an invalid resource");

    delete Resource;
    Resource = nullptr;

    return Plugin::success();
  }

  operator ResourceTy *() { return Resource; }

private:
  ResourceTy *Resource = nullptr;
};

template <typename ResourceTy>
class MPIResourcePool
    : GenericDeviceResourcePoolTy<MPIResourceRef<ResourceTy>> {
  using ResourceRefTy = MPIResourceRef<ResourceTy>;
  using ResourcePoolTy = GenericDeviceResourcePoolTy<ResourceRefTy>;

public:
  MPIResourcePool(GenericDeviceTy &Device, const char *ConfigEnvVar,
                  uint32_t DefNumResources = 32)
      : ResourcePoolTy(Device),
        InitialNumResources(ConfigEnvVar, DefNumResources) {}

  Error init() { return ResourcePoolTy::init(InitialNumResources.get()); }

  ResourceRefTy getResource() { return ResourcePoolTy::getResource(); }

  void releaseResource(ResourceRefTy Resource) {
    ResourcePoolTy::returnResource(Resource);
  }

private:
  UInt32Envar InitialNumResources;
};

// Device class
// =============================================================================
struct MPIDeviceTy : GenericDeviceTy {
  MPIDeviceTy(int32_t DeviceId, int32_t NumDevices, EventSystemTy &EventSystem)
      : GenericDeviceTy(DeviceId, NumDevices, MPIGridValues),
        MPIEventQueueManager(*this), EventSystem(EventSystem) {}

  Error initImpl(GenericPluginTy &Plugin) override {
    return MPIEventQueueManager.init();
  }

  Error deinitImpl() override { return MPIEventQueueManager.deinit(); }

  Error setContext() override { return Plugin::success(); }

  /// Load the binary image into the device and allocate an image object.
  Expected<DeviceImageTy *> loadBinaryImpl(const __tgt_device_image *TgtImage,
                                           int32_t ImageId) override {
    // Allocate and initialize the image object.
    MPIDeviceImageTy *Image = Plugin::get().allocate<MPIDeviceImageTy>();
    new (Image) MPIDeviceImageTy(ImageId, TgtImage);

    // Create a temporary file.
    char TmpFileName[] = "/tmp/tmpfile_XXXXXX";
    int TmpFileFd = mkstemp(TmpFileName);
    if (TmpFileFd == -1)
      return Plugin::error("Failed to create tmpfile for loading target image");

    // Open the temporary file.
    FILE *TmpFile = fdopen(TmpFileFd, "wb");
    if (!TmpFile)
      return Plugin::error("Failed to open tmpfile %s for loading target image",
                           TmpFileName);

    // Write the image into the temporary file.
    size_t Written = fwrite(Image->getStart(), Image->getSize(), 1, TmpFile);
    if (Written != 1)
      return Plugin::error("Failed to write target image to tmpfile %s",
                           TmpFileName);

    // Close the temporary file.
    int Ret = fclose(TmpFile);
    if (Ret)
      return Plugin::error("Failed to close tmpfile %s with the target image",
                           TmpFileName);

    // Load the temporary file as a dynamic library.
    std::string ErrMsg;
    DynamicLibrary DynLib =
        DynamicLibrary::getPermanentLibrary(TmpFileName, &ErrMsg);

    // Check if the loaded library is valid.
    if (!DynLib.isValid())
      return Plugin::error("Failed to load target image: %s", ErrMsg.c_str());

    // Save a reference of the image's dynamic library.
    Image->setDynamicLibrary(DynLib);

    return Image;
  }

  // Data management
  // ===========================================================================
  void *allocate(size_t Size, void *, TargetAllocTy Kind) override {
    if (Size == 0)
      return nullptr;

    void *BufferAddress = nullptr;
    Error Err = Error::success();
    EventPtr Event = nullptr;

    switch (Kind) {
    case TARGET_ALLOC_DEFAULT:
    case TARGET_ALLOC_DEVICE:
      Event = EventSystem.createEvent(OriginEvents::allocateBuffer, DeviceId,
                                      &BufferAddress);

      if (!Event) {
        Err = Plugin::error("Failed to create alloc event with size %z", Size);
        break;
      }

      Event->wait();
      Err = std::move(Event->getError());
      break;
    case TARGET_ALLOC_HOST:
    case TARGET_ALLOC_SHARED:
      Err = Plugin::error("Incompatible memory type %d", Kind);
      break;
    }

    if (Err) {
      REPORT("Failed to allocate memory: %s\n",
             toString(std::move(Err)).c_str());
      return nullptr;
    }

    return BufferAddress;
  }

  int free(void *TgtPtr, TargetAllocTy Kind) override {
    if (TgtPtr == nullptr)
      return OFFLOAD_SUCCESS;

    Error Err = Error::success();
    EventPtr Event = nullptr;

    switch (Kind) {
    case TARGET_ALLOC_DEFAULT:
    case TARGET_ALLOC_DEVICE:
      Event =
          EventSystem.createEvent(OriginEvents::deleteBuffer, DeviceId, TgtPtr);

      if (!Event) {
        Err = Plugin::error("Failed to create delete event");
        break;
      }

      Event->wait();
      Err = std::move(Event->getError());
      break;
    case TARGET_ALLOC_HOST:
    case TARGET_ALLOC_SHARED:
      Err = createStringError(inconvertibleErrorCode(),
                              "Incompatible memory type %d", Kind);
      break;
    }

    if (Err) {
      REPORT("Failed to free memory: %s\n", toString(std::move(Err)).c_str());
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

  // Data transfer
  // ===========================================================================
  Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                       AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    auto &Queue = getQueue(AsyncInfoWrapper);

    auto Event = EventSystem.createEvent(OriginEvents::submit, DeviceId, HstPtr,
                                         TgtPtr, Size);

    if (Event)
      return Plugin::error("Failed to create submit event");

    Queue->push_back(Event);

    return Plugin::success();
  }

  Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    auto &Queue = getQueue(AsyncInfoWrapper);

    auto Event = EventSystem.createEvent(OriginEvents::retrieve, DeviceId,
                                         HstPtr, TgtPtr, Size);

    if (Event)
      return Plugin::error("Failed to create retrieve event");

    Queue->push_back(Event);

    return Plugin::success();
  }

  Error dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstDev,
                         void *DstPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    auto &Queue = getQueue(AsyncInfoWrapper);

    auto Event =
        EventSystem.createEvent(OriginEvents::exchange, DeviceId, SrcPtr,
                                DstDev.getDeviceId(), DstPtr, Size);

    if (Event)
      return Plugin::error("Failed to create exchange event");

    Queue->push_back(Event);

    return Plugin::success();
  }

  // Target execution
  // ===========================================================================
  Expected<GenericKernelTy *>
  constructKernelEntry(const __tgt_offload_entry &KernelEntry,
                       DeviceImageTy &Image) override {

                       }

  // External event management
  // ===========================================================================
  Error createEventImpl(void **EventStoragePtr) override {
    if (!EventStoragePtr)
      return Plugin::error("Received invalid event storage pointer");

    EventPtr *NewEvent = MPIEventManager.getResource();
    if (NewEvent == nullptr)
      return Plugin::error("Could not allocate a new synchronization event");

    *EventStoragePtr = reinterpret_cast<void *>(NewEvent);

    return Plugin::success();
  }

  Error destroyEventImpl(void *Event) override {
    if (!Event)
      return Plugin::error("Received invalid event pointer");

    MPIEventManager.releaseResource(reinterpret_cast<EventPtr *>(Event));

    return Plugin::success();
  }

  Error recordEventImpl(void *Event,
                        AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    if (!Event)
      return Plugin::error("Received invalid event pointer");

    auto &Queue = getQueue(AsyncInfoWrapper);

    if (Queue->empty())
      return Plugin::success();

    auto &SyncEvent = *reinterpret_cast<EventPtr *>(Event);
    *SyncEvent = OriginEvents::sync(Queue->back());

    return Plugin::success();
  }

  Error waitEventImpl(void *Event,
                      AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    if (!Event)
      return Plugin::error("Received invalid event pointer");

    auto &SyncEvent = *reinterpret_cast<EventPtr *>(Event);

    auto &Queue = getQueue(AsyncInfoWrapper);

    Queue->push_back(SyncEvent);

    return Plugin::success();
  }

  Error syncEventImpl(void *Event) override {
    if (!Event)
      return Plugin::error("Received invalid event pointer");

    auto &SyncEvent = *reinterpret_cast<EventPtr *>(Event);
    SyncEvent->wait();

    return std::move(SyncEvent->getError());
  }

  // Asynchronous queue management
  // ===========================================================================
  Error synchronizeImpl(__tgt_async_info &AsyncInfo) override {
    auto Queue = reinterpret_cast<MPIEventQueue *>(AsyncInfo.Queue);

    for (auto &Event : *Queue) {
      Event->wait();

      if (auto &Error = Event->getError(); Error)
        return Plugin::error("Event failed during synchronization. %s\n",
                             toString(std::move(Error)));
    }

    MPIEventQueueManager.releaseResource(Queue);
    AsyncInfo.Queue = nullptr;

    return Plugin::success();
  }

  // Device environment
  // NOTE: not applicable to MPI.
  // ===========================================================================
  virtual bool shouldSetupDeviceEnvironment() const override { return false; };

  // Device memory limits
  // NOTE: not applicable to MPI.
  // ===========================================================================
  Error getDeviceStackSize(uint64_t &Value) override {
    Value = 0;
    return Plugin::success();
  }

  Error setDeviceStackSize(uint64_t Value) override {
    return Plugin::success();
  }

  Error getDeviceHeapSize(uint64_t &Value) override {
    Value = 0;
    return Plugin::success();
  }

  Error setDeviceHeapSize(uint64_t Value) override { return Plugin::success(); }

  // Device interoperability
  // NOTE: not supported by MPI right now.
  // ===========================================================================
  Error initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return Plugin::error("initAsyncInfoImpl not supported");
  }

  Error initDeviceInfoImpl(__tgt_device_info *DeviceInfo) override {
    return Plugin::error("initDeviceInfoImpl not supported");
  }

  // Debugging & Logging
  // ===========================================================================
  Error printInfoImpl() override {
    printf("    MPI proxy plugin\n");
    EventSystem.printInfo();

    return Plugin::success();
  }

private:
  MPIEventQueuePtr &getQueue(AsyncInfoWrapperTy &AsyncInfoWrapper) {
    MPIEventQueuePtr &Queue = AsyncInfoWrapper.getQueueAs<MPIEventQueuePtr>();
    if (Queue)
      Queue = MPIEventQueueManager.getResource();
    return Queue;
  }

  MPIResourcePool<MPIEventQueue> MPIEventQueueManager;
  MPIResourcePool<EventPtr> MPIEventManager;
  EventSystemTy &EventSystem;

  /// Grid values for the MPI plugin.
  static constexpr GV MPIGridValues = {
      1, // GV_Slot_Size
      1, // GV_Warp_Size
      1, // GV_Max_Teams
      1, // GV_SimpleBufferSize
      1, // GV_Max_WG_Size
      1, // GV_Default_WG_Size
  };
};

/// Class implementing the MPI plugin.
struct MPIPluginTy : GenericPluginTy {
  MPIPluginTy() : GenericPluginTy() {
    EventSystem.initialize();
    GenericPluginTy::init(EventSystem.getNumWorkers(),
                          new MPIGlobalHandlerTy());
  }

  ~MPIPluginTy() override {
    EventSystem.deinitialize();
    MPIPluginTy::~MPIPluginTy();
  }

  /// This class should not be copied.
  MPIPluginTy(const MPIPluginTy &) = delete;
  MPIPluginTy(MPIPluginTy &&) = delete;

  /// Get the ELF code to recognize the compatible binary images.
  uint16_t getMagicElfBits() const override { return TARGET_ELF_ID; }

  /// Create a MPI device instance for the given ID.
  MPIDeviceTy &createDevice(int32_t DeviceId) override {
    return Devices.emplace_back(DeviceId, getNumDevices(), EventSystem);
  }

  bool isDataExchangable(int32_t SrcDeviceId, int32_t DstDeviceId) override {
    return isValidDeviceId(SrcDeviceId) && isValidDeviceId(DstDeviceId);
  }

  /// All images (ELF-compatible) should be compatible with this plugin.
  Expected<bool> isImageCompatible(__tgt_image_info *Info) const override {
    return true;
  }

private:
  EventSystemTy EventSystem;
  llvm::SmallVector<MPIDeviceTy, 8> Devices;
};

/// Static plugin interface.
static MPIPluginTy *MPIPlugin = nullptr;

Error Plugin::init() {
  if (MPIPlugin)
    return Plugin::error("MPI plugin is already initialized");

  MPIPlugin = new MPIPluginTy;

  if (MPIPlugin)
    return Plugin::error("Could not allocate new MPI plugin instance");

  return Plugin::success();
}

Error Plugin::deinit() {
  if (!Plugin::isActive())
    return Plugin::error("MPI plugin is not initialized");

  if (MPIPlugin)
    return Plugin::error("No MPI plugin was found");

  delete MPIPlugin;

  return Plugin::success();
}

GenericPluginTy &Plugin::get() {
  assert(Plugin::isActive() && "MPI plugin was not initialized");
  return *MPIPlugin;
}

template <typename... ArgsTy>
Error Plugin::check(int32_t ErrorCode, const char *ErrFmt, ArgsTy... Args) {
  if (ErrorCode == 0)
    return Error::success();

  return createStringError<ArgsTy..., const char *>(
      inconvertibleErrorCode(), ErrFmt, Args...,
      std::to_string(ErrorCode).data());
}

} // namespace llvm::omp::target::plugin
