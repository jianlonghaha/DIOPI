#include "Handle.h"
#include "DeviceThreadHandles.h"
// #include "torch_musa/csrc/core/MUSAStream.h"
#include <torch/torch.h> 
#include "Utils.h"
#include <iostream>
#define TORCH_MUSA_CHECK(EXPR)                                       \
  do {                                                               \
    const musaError_t __err = EXPR;                                  \
    if (__err != musaSuccess) {                                      \
      TORCH_CHECK(false, "MUSA error: ", musaGetErrorString(__err)); \
    }                                                                \
  } while (0)
namespace  at{ 
namespace  impl{ 
namespace  musa{ 
namespace {

void CreateMuDNNHandle(mudnnHandle_t* handle) {
  TORCH_CHECK(handle, "Handle pointer is no-nullptr");
  int device;
  TORCH_MUSA_CHECK(musaGetDevice(&device));
  TORCH_CHECK(device >= 0);
  std::cout << "==================== " << device << " ==========================\n";

  *handle = new impl::musa::muHandle(device);
}

void DestroyMuDNNHandle(mudnnHandle_t /*handle*/) {
  // this is because of something dumb in the ordering of
  // destruction. Sometimes atexit, the musa context (or something)
  // would already be destroyed by the time this gets destroyed. It
  // happens in fbcode setting. Not destroy the handle as a workaround.
}

using MudnnPoolType = at::impl::musa::DeviceThreadHandlePool<
    mudnnHandle_t,
    CreateMuDNNHandle,
    DestroyMuDNNHandle>;

} // namespace

::musa::dnn::Handle& GetMudnnHandle() {
  int device;
  TORCH_MUSA_CHECK(musaGetDevice(&device));
  std::cout << "==================== " << device << " ==Handle.cpp========================\n";

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.
  static auto pool = std::make_shared<MudnnPoolType>();
  thread_local std::unique_ptr<MudnnPoolType::PoolWindow> myPoolWindow(
      pool->NewPoolWindow());

  mudnnHandle_t handle = myPoolWindow->reserve(device);
  // handle->SetStream(c10::musa::getCurrentMUSAStream());
  return *handle;
}
}
}
} // namespace at
// } // namespace at
