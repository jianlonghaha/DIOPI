#include <ATen/core/Tensor.h>
#include <ATen/ATen.h>
#include <ATen/core/Scalar.h>

#include "./common/common.hpp"

// namespace at {
// namespace musa {
//source_torch_musa 
// extern  Tensor& AddTensorOut(const Tensor&, const Tensor&, Scalar const&, Tensor&);
// extern  Tensor& MulTensorOut(const Tensor&, const Tensor&, Tensor&);
// extern  Tensor& MulScalarOut(const Tensor&, const Tensor&, Tensor&);
// }
// }
namespace musa_torch {
using at::Tensor;
using at::ScalarType;
using at::Scalar;
}
