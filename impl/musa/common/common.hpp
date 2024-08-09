#ifndef IMPL_MUSA_COMMON_COMMON_HPP_
#define IMPL_MUSA_COMMON_COMMON_HPP_

#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <ATen/ATen.h>
#include <cassert>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "../musa_pytorch.h"
#include "MUSAHooksInterface.h"
#include <ATen/core/TensorBase.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/Scalar.h>

#define DEBUG false
namespace  at{
namespace impl {
namespace musa {


inline void alpha_check(const ScalarType dtype, const Scalar& alpha) {
  TORCH_CHECK(! alpha.isBoolean() || dtype == ScalarType::Bool,
              "Boolean alpha only supported for Boolean results.");
  TORCH_CHECK(isFloatingType(dtype) || isComplexType(dtype)
              || alpha.isIntegral(true),
              "For integral input tensors, argument alpha must not be a floating point number.");
  TORCH_CHECK(isComplexType(dtype) || !alpha.isComplex(),
              "For non-complex input tensors, argument alpha must not be a complex number.")
}


inline bool is_scalar(const at::Tensor& tensor) {
  return tensor.numel() == 1 && tensor.dim() == 0;
}


const at::MUSAHooksInterface& hooks = at::detail::getMUSAHooks();
// REGISTER_MUSA_HOOKS(hooks);
inline bool isInt(const diopiScalar_t* scalar) { return scalar->stype <= 7; }
inline bool isFloat(const diopiScalar_t* scalar) { return scalar->stype > 7; }

c10::Scalar build_musatorch_scalar(const diopiScalar_t* scalar) {
    if (scalar == nullptr) {
        throw std::invalid_argument("scalar is nullptr, temporarily unsupported");
    }
    if (DEBUG) {
        printf("scalar type is %d\n", static_cast<int>(scalar->stype));
    }
    if (isInt(scalar)) {
        int64_t ival = scalar->ival;
        return c10::Scalar(ival);
    } else {
        float fval = scalar->fval;
        return c10::Scalar(fval);
    }
}
c10::ScalarType get_musatorch_type(diopiDtype_t dt) {
    switch (dt) {
        case diopi_dtype_bool:
            return c10::ScalarType::Bool;
        case diopi_dtype_uint8:
            return c10::ScalarType::Byte;
        case diopi_dtype_int8:
            return c10::ScalarType::Char;
        case diopi_dtype_int16:
            return c10::ScalarType::Short;
        case diopi_dtype_float32:
            return c10::ScalarType::Float;
        case diopi_dtype_int32:  
            return c10::ScalarType::Int;
        case diopi_dtype_int64:  
            return c10::ScalarType::Long;
        case diopi_dtype_float16: 
            return c10::ScalarType::Half;
        case diopi_dtype_float64: 
            return c10::ScalarType::Double;   
        default:
            throw std::invalid_argument("Unsupported diopi dtype");
    }
}



template <typename T>
at::Tensor build_musatorch_tensor(T tensor) {
    if (DEBUG) {
        printf("tensor building... \n");
    }
    if (tensor == nullptr) {
        if (DEBUG) {
            printf("tensor is nullptr\n");
        }
        std::cout << "==================// 创建空Tensor=======================\n";
        return at::empty({0}, at::TensorOptions().dtype(at::kFloat)); // 创建空Tensor
    }
    diopiSize_t dsize;
    diopiError_t err = diopiGetTensorShape(tensor, &dsize);
    if (err != diopiSuccess) {
        throw std::runtime_error("bad tensor shape");
    }
    if (DEBUG) {
        printf("tensor dsize len is %ld\n", dsize.len);
    }
    void* data = nullptr;
    err = diopiGetTensorData(const_cast<diopiTensorHandle_t>(tensor), &data);
    if (err != diopiSuccess) {
        throw std::runtime_error("bad tensor data");
    }
    if (DEBUG) {
        printf("tensor ptr is %p\n", data);
    }
    diopiDtype_t dtype;
    err = diopiGetTensorDtype(tensor, &dtype);
    if (err != diopiSuccess) {
        throw std::runtime_error("bad tensor datatype");
    }
    if (DEBUG) {
        printf("tensor dtype is %d\n", static_cast<int>(dtype));
    }
    auto scalar_type = get_musatorch_type(dtype);
    diopiSize_t shape;
    err = diopiGetTensorShape(tensor, &shape);
    if (err != diopiSuccess) {
        throw std::runtime_error("bad tensor shape");
    }
    std::vector<int64_t> tensor_sizes(shape.data, shape.data + shape.len);
    if (dsize.len == 0) {
        tensor_sizes.push_back(1);
    }
    diopiSize_t stride;
    err = diopiGetTensorStride(tensor, &stride);
    if (err != diopiSuccess) {
        throw std::runtime_error("bad tensor stride");
    }
    std::vector<int64_t> tensor_strides(stride.data, stride.data + stride.len);
    if (dsize.len == 0) {
        tensor_strides.push_back(1);
    }
    // auto options = at::TensorOptions().dtype(scalar_type).device(at::kPrivateUse1);
    auto options = at::TensorOptions().dtype(scalar_type).device(at::kPrivateUse1);
    // auto options = at::TensorOptions().dtype(scalar_type).device(at::kCPU);
    // auto options = at::TensorOptions().dtype(scalar_type).device(at::kXPU);
    // return at::from_blob(data, tensor_sizes, tensor_strides,options.requires_grad(c10::nullopt)); 
    // at::TensorOptions options = at::TensorOptions().dtype(at::kFloat).device(at::kPrivateUse1, 0);
    return at::from_blob(data, tensor_sizes, tensor_strides,options); // 从已有数据创建Tensor
}

}  // namespace musa
}  // namespace impl
}
#endif  // IMPL_MUSA_COMMON_COMMON_HPP_