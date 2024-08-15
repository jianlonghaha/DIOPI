#ifndef MUSA_MM_HPP
#define MUSA_MM_HPP
#include "../common/common.hpp"
#include "../common/Utils.h"
#include <mudnn.h>
#include <ATen/ATen.h>
#include <c10/core/Scalar.h> 
#include <iostream> 

namespace at{
namespace impl {
namespace musa{
 



void MmCall(
    const at::Tensor& l,
    const at::Tensor& r,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& out,
    bool is_batch = false,
    const at::Scalar& alpha = 1,
    const at::Scalar beta = 0,
    const at::Scalar gama = 0) {

  std::cout << "l shape: " << l.sizes() << std::endl;
  std::cout << "r shape: " << r.sizes() << std::endl;
  std::cout << "out shape: " << out.sizes() << std::endl;

  if (l.numel() == 0 || r.numel() == 0) {
    out.zero_();
    return;
  }

  at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();
  bool trans_l = at::impl::musa::IsTranspose(l);
  bool trans_r = at::impl::musa::IsTranspose(r);

  at::Tensor contiguous_l;
  at::Tensor contiguous_r;

  at::impl::musa::muTensor  lmt = trans_l ? at::impl::musa::CreateMUTensor(l.transpose(-2, -1))
                     : at::impl::musa::CreateMUTensor(at::impl::musa::ContiguousRef(l, contiguous_l,c10::MemoryFormat::Contiguous));
  at::impl::musa::muTensor  rmt = trans_r ? at::impl::musa::CreateMUTensor(r.transpose(-2, -1))
                     : at::impl::musa::CreateMUTensor(at::impl::musa::ContiguousRef(r, contiguous_r,c10::MemoryFormat::Contiguous));
  at::impl::musa::muTensor  rst = at::impl::musa::CreateMUTensor(out);

  // at::Tensor cpuTensor = deviceTensor.toCPU(); 
  // auto lmt_cpu = lmt.to(at::kCPU);
  // size_t numElements = lmt.numel(); // 假设 numel() 方法返回元素总数

  // std::cout << "=================hello,lmt cpu========="<<cpuTensor<<"==============\n";
  // std::cout << "=================hello,rmt==========="<<rmt<<"================\n";
  // std::cout << "=================hello,rst============="<<lmt<<"==============\n";




  if (is_batch) {
    ::musa::dnn::BatchMatMul b_mm;
    CHECK_MUDNN_STATUS(
        b_mm.SetComputeMode(at::impl::musa::GetComputeModeFromCtx(l.scalar_type())),
        "SetComputeMode");
    CHECK_MUDNN_STATUS(b_mm.SetTranspose(trans_l, trans_r), "SetTranspose");
    CHECK_MUDNN_STATUS(b_mm.Run(h, rst, lmt, rmt, InternalMemAlloc), "Run");
    // CHECK_MUDNN_STATUS(b_mm.Run(h, rst, lmt, rmt, at::impl::musa::InternalMemAlloc), "Run");

    // CHECK_MUDNN_STATUS(b_mm.Run(h, rst, lmt, rmt), "Run");
    std::cout << "=================hello,is_batch========================\n";


  } else {
    std::cout << "=================hello,is_not_batch========================\n";
    ::musa::dnn::MatMul mm;
    CHECK_MUDNN_STATUS(
        mm.SetComputeMode(at::impl::musa::GetComputeModeFromCtx(l.scalar_type())),
        "SetComputeMode");
    CHECK_MUDNN_STATUS(mm.SetTranspose(trans_l, trans_r), "SetTranspose");
    // not support broadcast
    if (bias.has_value()) {
      std::cout << "bias shape: " << bias->sizes() << std::endl;

      TORCH_INTERNAL_ASSERT(bias->dim() == 1, "bias must be 1d tensor\n");
      // auto bmt = CreateMUTensor(bias.value());
      auto bmt = at::impl::musa::CreateMUTensor(bias.value());

      CHECK_MUDNN_STATUS(mm.SetAlpha(alpha.to<double>()), "SetAlpha");
      CHECK_MUDNN_STATUS(mm.SetBeta(beta.to<double>()), "SetBeta");
      CHECK_MUDNN_STATUS(mm.SetGamma(gama.to<double>()), "SetGamma");
      CHECK_MUDNN_STATUS(
          mm.RunWithBiasAdd(h, rst, lmt, rmt, bmt, at::impl::musa::InternalMemAlloc),
          // mm.RunWithBiasAdd(h, rst, lmt, rmt, bmt),

          "RunWithBiasAdd");
    } else {
      CHECK_MUDNN_STATUS(mm.SetAlpha(alpha.to<double>()), "SetAlpha");
      CHECK_MUDNN_STATUS(mm.SetBeta(beta.to<double>()), "SetBeta");
      CHECK_MUDNN_STATUS(mm.Run(h, rst, lmt, rmt, at::impl::musa::InternalMemAlloc), "Run");
      // CHECK_MUDNN_STATUS(mm.Run(h, rst, lmt, rmt), "Run");
    }
  }
}


at::Tensor& MmOut(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  std::cout << "self shape: " << self.sizes() << std::endl;
  std::cout << "mat2 shape: " << mat2.sizes() << std::endl;
  std::cout << "out shape: " << out.sizes() << std::endl;
  std::cout << "=================hello,MmOut001========================\n";

  TORCH_CHECK(
      self.dim() == 2 && mat2.dim() == 2 && self.size(1) == mat2.size(0),
      "self and mat2 must be a matrix and self_shape[1] must equal to "
      "mat2_shape[0]");
  MmCall(self, mat2, c10::nullopt, out, false, 1, 0);
  std::cout << "=================hello,MmOut========================\n";

  return out;
}

// at::Tensor Mm(const at::Tensor& self, const at::Tensor& mat2) {
//   at::Tensor result = at::empty(
//       {self.size(0), mat2.size(1)},
//       self.options().memory_format(at::MemoryFormat::Contiguous));
//   MmOut(self, mat2, result);
//   std::cout << "=================hello,Mm========================\n";

//   return result;
// }


// Tensor Bmm(const Tensor& self, const Tensor& mat2) {
//   Tensor result = at::empty(
//       {self.size(0), self.size(1), mat2.size(2)},
//       self.options().memory_format(at::MemoryFormat::Contiguous));
//   BmmOut(self, mat2, result);
//   return result;
// }


at::Tensor& BmmOut(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  // const auto device_guard = c10::musa::MUSAGuard(self.device());
  TORCH_CHECK(self.dim() == 3 && mat2.dim() == 3, "self must be a 3D matrix");
  TORCH_CHECK(
      self.size(0) == mat2.size(0) && self.size(2) == mat2.size(1),
      "self_shape[0] must equal to mat2_shape[0], and self_shape[2] "
      "must equal to mat2_shape[1]");
 
    
  MmCall(self, mat2, c10::nullopt, out, true);
  std::cout << "=================hello,BmmOut========================\n";
  return out;
}

}
}
}
#endif // MUSA_MM_HPP
