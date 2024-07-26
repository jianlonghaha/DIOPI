#include <diopi/functions.h>
#include <math.h>
#include <cstring>
#include "../musa_pytorch.h"
#include "../common/common.hpp"
#include <mudnn.h>
#include "../common/Utils.h"
#include "../common/Handle.h"
// #include <ATen/Functions.h>
// #include <torch/library.h>
namespace impl {
namespace musa {
using BINARY_MODE = ::musa::dnn::Binary::Mode;
extern "C" {
static const char* name = "MUSADevice";
DIOPI_RT_API const char* diopiGetVendorName() { return name; }
DIOPI_API diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                const diopiScalar_t* alpha) {
            musa_torch::Tensor _out = at::impl::musa::build_musatorch_tensor(out);
            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            musa_torch::Tensor _other = at::impl::musa::build_musatorch_tensor(other);
            musa_torch::Scalar _alpha = at::impl::musa::build_musatorch_scalar(alpha);
            // 如果 self 或 other 张量为空，则直接返回diopiDtypeNotSupported
            // if (C10_UNLIKELY(_input.numel() == 0 || _other.numel() == 0)) {
            //      return diopiDtypeNotSupported;
            //      }
            // // 检查张量是否在同一个设备上
            // if ((!at::impl::musa::is_scalar(_other) && _other.device().is_cpu()) ||
            //     (!at::impl::musa::is_scalar(_input) && _input.device().is_cpu())) {
            //     TORCH_CHECK(
            //         false,
            //         "Expected all tensors to be on the same device, but "
            //         "found at least two devices, ",
            //         _input.device().type(),
            //         " and ",
            //         _other.device().type(),
            //         "!")
            // };
            if (at::impl::musa::is_musa(_input) && at::impl::musa::is_musa(_other) &&
                _input.device().index() != _other.device().index()) {
                TORCH_CHECK(
                    false,
                    "Expected all tensors to be on the same device, but "
                    "found at least two devices, ",
                    _input.device(),
                    " and ",
                    _other.device(),
                    "!")
            };     
           // 获取MUDNN处理句柄，处理add操作
            BINARY_MODE m = BINARY_MODE::ADD;           
            at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();
            at::impl::musa::muTensor musa_self = at::impl::musa::CreateMUTensor(_input);
            at::impl::musa::muTensor musa_other = at::impl::musa::CreateMUTensor(_other);
            at::impl::musa::muTensor musa_out = at::impl::musa::CreateMUTensor(_out);
            ::musa::dnn::Binary bop;
            bop.SetMode(m);
            bop.Run(h, musa_out, musa_self, musa_other);
            return diopiSuccess;
}

DIOPI_API diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
            impl::musa::diopiAdd(ctx, input, input, other, alpha);
            return diopiSuccess;
}

DIOPI_API diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                                      const diopiScalar_t* alpha) {
            impl::musa::diopiAdd(ctx, out, input, input, alpha);
            return diopiSuccess;
}

DIOPI_API diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
            musa_torch::Tensor _out = at::impl::musa::build_musatorch_tensor(out);
            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            musa_torch::Tensor _other = at::impl::musa::build_musatorch_tensor(other);          
            BINARY_MODE m = BINARY_MODE::MUL;

            at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();
            at::impl::musa::muTensor musa_self = at::impl::musa::CreateMUTensor(_input);
            at::impl::musa::muTensor musa_other = at::impl::musa::CreateMUTensor(_other);
            at::impl::musa::muTensor musa_out = at::impl::musa::CreateMUTensor(_out);


         

            ::musa::dnn::Binary bop;
            bop.SetMode(m);
            bop.Run(h, musa_out, musa_self, musa_other);
            return diopiSuccess;
}
DIOPI_API diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
            impl::musa::diopiAddScalar(ctx, input, input, other, alpha);
            return diopiSuccess;
}


}   //extern C
}  // namespace musa
}  // namespace impl
