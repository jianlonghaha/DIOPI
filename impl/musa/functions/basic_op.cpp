#include <diopi/functions.h>
#include <math.h>
#include <cstring>
#include <mudnn.h>
#include <ATen/ATen.h>
#include <c10/core/Scalar.h> 
// #include <c10/util/DimVector.h>
#include <c10/core/Device.h>
#include <vector>
#include <cstddef> // for size_t
#include <ATen/WrapDimUtils.h>


#include "../musa_pytorch.h"
#include "../common/Context.h"
#include "../common/common.hpp"
#include "../common/Utils.h"
#include "../common/Handle.h"
#include "../common/Matmul.hpp" 
#include "../common/Reduce.hpp"



namespace impl {
namespace musa {
    
using BINARY_MODE = ::musa::dnn::Binary::Mode;
using UNARY_MODE =  ::musa::dnn::Unary::Mode;
using SOFTMAX_MODE  = ::musa::dnn::Softmax::Mode;
using REDUCE_MODE = ::musa::dnn::Reduce::Mode;
extern "C" {
static const char* name = "MUSADevice";
DIOPI_RT_API const char* diopiGetVendorName() { return name; }
// DIOPI_API diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
//                                 const diopiScalar_t* alpha){                                
//             std::cout << "=================hello,sub00========================\n";
//             musa_torch::Tensor _out = at::impl::musa::build_musatorch_tensor(out);
//             musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
//             musa_torch::Tensor _other = at::impl::musa::build_musatorch_tensor(other);
//             // musa_torch::Scalar _alpha = at::impl::musa::build_musatorch_scalar(alpha);
//             std::cout << "=================hello,sub01========================\n";
//             BINARY_MODE m = BINARY_MODE::SUB;           
//             at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();
//             at::impl::musa::muTensor musa_self = at::impl::musa::CreateMUTensor(_input);
//             at::impl::musa::muTensor musa_other = at::impl::musa::CreateMUTensor(_other);
//             at::impl::musa::muTensor musa_out = at::impl::musa::CreateMUTensor(_out);
//             ::musa::dnn::Binary bop;
//             bop.SetMode(m);
//             bop.Run(h, musa_out, musa_self, musa_other);
//             std::cout << "=================hello,sub02========================\n";
//             return diopiSuccess;                        
//                                 }
// DIOPI_API diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim){            
// }


void PrintTensorValues(const at::Tensor& tensor) {
    std::cout << "================开始===============\n";
    at::Tensor tensor_cpu = tensor.cpu();
    auto sizes = tensor_cpu.sizes();
    int batch_size = sizes[0];
    int rows = sizes[1];
    int cols = sizes[2];
    for (int b = 0; b < batch_size; ++b) {
        std::cout << "Batch " << b << ":\n";
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                std::cout << tensor_cpu[b][r][c].item<float>() << " ";
            }
            std::cout << "\n"; 
        }
        std::cout << "\n"; 
    }
    std::cout << "================结束===============\n";
}


DIOPI_API diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                const diopiScalar_t* alpha) {         
            std::cout << "================diopiAdd::out::======="<<out<<"=================\n";         
                        
            musa_torch::Tensor _out = at::impl::musa::build_musatorch_tensor(out);
            std::cout << "================diopidiopiAdd__out::scalar_type======="<<_out.scalar_type()<<"=================\n";         
            std::cout << "================diopidiopiAdd__out::device======="<<_out.device()<<"=================\n";         

            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            musa_torch::Tensor _other = at::impl::musa::build_musatorch_tensor(other);
            musa_torch::Scalar _alpha = at::impl::musa::build_musatorch_scalar(alpha);
            PrintTensorValues(_out);
            PrintTensorValues(_input);
            PrintTensorValues(_other);





            std::cout << "================diopidiopiAdd_input::torch::======="<<_input.scalar_type()<<"=================\n";         
            std::cout << "=================diopidiopiAdd_other::torch::============="<<_other.scalar_type()<<"===========\n";    
            std::cout << "=================input.device: =================" << _input.device() << "\n";
            std::cout << "=================out.device: =================" << _out.device() << "\n";
            // TORCH_CHECK(
            //         _input.scalar_type() == _other.scalar_type(),
            //          "diopiAdd input scalar type must the same");
            // 如果 self 或 other 张量为空，则直接返回diopiDtypeNotSupported
            if (C10_UNLIKELY(_input.numel() == 0 || _other.numel() == 0)) {
                 return diopiDtypeNotSupported;
                 }
            // 检查张量是否在同一个设备上
            if ((!at::impl::musa::is_scalar(_other) && _other.device().is_cpu()) ||
                (!at::impl::musa::is_scalar(_input) && _input.device().is_cpu())) {
                TORCH_CHECK(
                    false,
                    "Expected all tensors to be on the same device, but "
                    "found at least two devices, ",
                    _input.device().type(),
                    " and ",
                    _other.device().type(),
                    "!")
            };
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
            musa_torch::ScalarType common_dtype = at::result_type(_input, _other);
            musa_torch::Tensor common_input = _input.to(common_dtype);  
            musa_torch::Tensor common_other=_other.to(common_dtype);
            BINARY_MODE m = BINARY_MODE::ADD;           
            at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();
            at::impl::musa::muTensor musa_input = at::impl::musa::CreateMUTensor(common_input);
            at::impl::musa::muTensor musa_other = at::impl::musa::CreateMUTensor(common_other);
            at::impl::musa::muTensor musa_out = at::impl::musa::CreateMUTensor(_out);
            ::musa::dnn::Binary bop;
            CHECK_MUDNN_STATUS(bop.SetMode(m),"SetMode");
            CHECK_MUDNN_STATUS(bop.Run(h, musa_out, musa_input, musa_other),"Run diopiAddScalar");
            // CHECK_MUDNN_STATUS(uop.Run(h, musa_out, musa_input),"Run diopiAddScalar");

            // std::cout << "=================hello,add:musa_out======"<< musa_out.item()<<"=================\n";
            return diopiSuccess;
}

DIOPI_API diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                                      const diopiScalar_t* alpha){
            std::cout << "================diopiAddScalar come in  !=================\n";         
            musa_torch::Tensor _out = at::impl::musa::build_musatorch_tensor(out);
            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            musa_torch::Scalar _other = at::impl::musa::build_musatorch_scalar(other);
            musa_torch::Scalar _alpha = at::impl::musa::build_musatorch_scalar(alpha); 
           

            // 测试用，先形成tensor 在进行torch对比用
            at::Tensor other_tensor;
            if (_other.isFloatingPoint()) {
                other_tensor = at::tensor(_other.toFloat());
            } else if (_other.isIntegral(/*includeBool=*/false)) {
                other_tensor = at::tensor(_other.toInt());
            } else if (_other.isBoolean()) {
                other_tensor = at::tensor(_other.toBool());
            } else {
                throw std::runtime_error("Unsupported scalar type");
            }
            std::cout << "================diopiAddScalar_input::torch::======="<<_input.scalar_type()<<"=================\n";         
            std::cout << "=================diopiAddScalar_other::torch::============="<<other_tensor.scalar_type()<<"===========\n";  
            // TORCH_CHECK(
            //         // _input.dtype().name() == other_tensor.dtype().name(), "diopiAddScalar input scalar type must the same");
            //         _input.scalar_type() == other_tensor.scalar_type(), "diopiAddScalar input scalar type must the same");


// 公共类型转换
            musa_torch::ScalarType common_dtype = at::result_type(_input, _other);
            musa_torch::Tensor common_input = _input.to(common_dtype);  
            musa_torch::Scalar common_other;
            if (common_dtype == musa_torch::ScalarType::Float) {
                std::cout << "=================else if (common_dtype == musa_torch::ScalarType::Float)=====================\n";
                common_other = _other.toFloat();
            } else if (common_dtype == musa_torch::ScalarType::Double) {
                std::cout << "=================else if (common_dtype == musa_torch::ScalarType::Double)=====================\n";
                common_other = _other.toDouble();
            } else if (common_dtype == musa_torch::ScalarType::Int) {
                std::cout << "=================else if (common_dtype == musa_torch::ScalarType::Int)=====================\n";
                common_other = _other.toInt();
            } else if (common_dtype == musa_torch::ScalarType::Long) {
                std::cout << "=================else if (common_dtype == musa_torch::ScalarType::Long)=====================\n";
                common_other = _other.toLong();
            } else {
                AT_ERROR("Unsupported ScalarType for Scalar conversion");
            }
            at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();
            // at::impl::musa::muTensor musa_self = at::impl::musa::CreateMUTensor(_input);
            at::impl::musa::muTensor musa_input = at::impl::musa::CreateMUTensor(common_input);
            at::impl::musa::muTensor musa_out = at::impl::musa::CreateMUTensor(_out);     
            ::musa::dnn::Unary uop;

            if (common_other.isFloatingPoint()) {
                CHECK_MUDNN_STATUS(uop.SetAlpha(common_other.toDouble()), "SetAlpha");
            } else if (common_other.isIntegral(false)) {
                CHECK_MUDNN_STATUS(uop.SetAlpha(common_other.toLong()), "SetAlpha");
            } else {
                AT_ERROR(
                    common_other.type(), " is not implemented");
            }
            CHECK_MUDNN_STATUS(uop.SetMode(UNARY_MODE::ADD), "SetMode");
            CHECK_MUDNN_STATUS(uop.Run(h, musa_out, musa_input),"Run diopiAddScalar");
            std::cout << "=================hello,diopiAddScalar========================\n";
            return diopiSuccess;
                                      }

DIOPI_API diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other){
            std::cout << "=================hello,diopiMulScalar========================\n";         
            musa_torch::Tensor _out = at::impl::musa::build_musatorch_tensor(out);
            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            musa_torch::Scalar _other = at::impl::musa::build_musatorch_scalar(other);


// 测试用，先形成tensor 在进行torch对比用
            at::Tensor other_tensor;
            if (_other.isFloatingPoint()) {
                other_tensor = at::tensor(_other.toFloat());
            } else if (_other.isIntegral(/*includeBool=*/false)) {
                // other_tensor = at::tensor(_other.toInt());
                other_tensor = at::tensor(_other.toLong());
            } else if (_other.isBoolean()) {
                other_tensor = at::tensor(_other.toBool());
            } else {
                throw std::runtime_error("Unsupported scalar type");
            }
            std::cout << "================diopiMulScalar::torch::_input======="<<_input.scalar_type()<<"=================\n";         
            std::cout << "=================diopiMulScalar::torch::_other============="<<other_tensor.scalar_type()<<"===========\n"; 
            // TORCH_CHECK(
            //         _input.scalar_type() == other_tensor.scalar_type(), "diopiMulScalar input scalar type must the same");
           


            musa_torch::ScalarType common_dtype = at::result_type(_input, _other);
            musa_torch::Tensor common_input = _input.to(common_dtype);  
            // musa_torch::Scalar common_other=_other.to(common_dtype);
            //  手动将标量 _other 转换为 common_dtype 类型
            musa_torch::Scalar common_other;
            if (common_dtype == musa_torch::ScalarType::Float) {
                common_other = _other.toFloat();
            } else if (common_dtype == musa_torch::ScalarType::Double) {
                common_other = _other.toDouble();
            } else if (common_dtype == musa_torch::ScalarType::Int) {
                common_other = _other.toInt();
            } else if (common_dtype == musa_torch::ScalarType::Long) {
                common_other = _other.toLong();
            } else {
                AT_ERROR("Unsupported ScalarType for Scalar conversion");
            }

            at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();
            at::impl::musa::muTensor musa_input = at::impl::musa::CreateMUTensor(common_input);
            at::impl::musa::muTensor musa_out = at::impl::musa::CreateMUTensor(_out);     
            ::musa::dnn::Unary uop;
            // auto other_scalar = _other.item();
            // auto other_scalar = common_other;
            if (common_other.isFloatingPoint()) {
                CHECK_MUDNN_STATUS(uop.SetAlpha(common_other.toDouble()), "SetAlpha");
            } else if (common_other.isIntegral(false)) {
                CHECK_MUDNN_STATUS(uop.SetAlpha(common_other.toLong()), "SetAlpha");
            } else {
                AT_ERROR(
                    common_other.type(), " is not implemented for broadcast in Binary");
            }
            CHECK_MUDNN_STATUS(uop.SetMode(UNARY_MODE::MUL), "SetMode");
            CHECK_MUDNN_STATUS(uop.Run(h, musa_out, musa_input),"Run diopiMulScalar");
            std::cout << "=================hello,diopiMulScalar========================\n";
            return diopiSuccess;
}


DIOPI_API diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
            musa_torch::Tensor _out = at::impl::musa::build_musatorch_tensor(out);
            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            musa_torch::Tensor _other = at::impl::musa::build_musatorch_tensor(other);   
            std::cout << "=======diopiMul,_input::torch::==========" << _input.dtype().name() << "========================\n";
            std::cout << "=======diopiMul,_input::torch::==========" << _other.dtype().name() << "========================\n";
            // std::cout << "=======_input.scalar_type()==========" << _input.scalar_type() << "========================\n";
            // TORCH_CHECK(
            //         _input.scalar_type() == _other.scalar_type(),
            //          "diopimul input scalar type must the same");    
            // at::impl::musa::alpha_check(common_dtype, _alpha);
            // musa_torch::Tensor common_self =
            //     at::impl::musa::ContiguousIfZeroInStrides(_input.to(common_dtype));
            //     std::cout << "=======at::impl::musa::ContiguousIfZeroInStrides(_input.to(common_dtype))=======================\n";
            // musa_torch::Tensor common_other =
            //     at::impl::musa::ContiguousIfZeroInStrides(_other.to(common_dtype));

            musa_torch::ScalarType common_dtype = at::result_type(_input, _other);
            std::cout << "=======musa_torch::ScalarType common_dtype=======================\n";
            musa_torch::Tensor common_input = _input.to(common_dtype);  
            std::cout << "=======_input=======================\n";
            musa_torch::Tensor common_other=_other.to(common_dtype);
            std::cout << "=======_other=======================\n";

            BINARY_MODE m = BINARY_MODE::MUL;
            at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();
            // at::impl::musa::muTensor musa_self = at::impl::musa::CreateMUTensor(_input);
            at::impl::musa::muTensor musa_input = at::impl::musa::CreateMUTensor(common_input);
            std::cout << "=======hello,musa_self==========" << _input.dtype().name() << "========================\n";
            // at::impl::musa::muTensor musa_other = at::impl::musa::CreateMUTensor(_other);
            at::impl::musa::muTensor musa_other = at::impl::musa::CreateMUTensor(common_other);
            at::impl::musa::muTensor musa_out = at::impl::musa::CreateMUTensor(_out);
            ::musa::dnn::Binary bop;
            bop.SetMode(m);
            bop.Run(h, musa_out, musa_input, musa_other);
            std::cout << "=================hello,mul========================\n";
            return diopiSuccess;
}



inline void CheckDimParams(const  musa_torch::Tensor& input, const int64_t dim) {
  int64_t dim_ = at::maybe_wrap_dim(dim, input.dim());
  int64_t input_dim = input.dim() > 0 ? input.dim() : 1;
  TORCH_CHECK(
      dim_ >= 0 && dim_ < input_dim,
      "dim must be non-negative and less than input dimensions");
}


DIOPI_API diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim){
            std::cout << "=================hello,diopiSoftmax========================\n";         
            musa_torch::Tensor _out = at::impl::musa::build_musatorch_tensor(out);
            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();

            auto contiguous_input = _input.contiguous();
            CheckDimParams(contiguous_input, dim);
            auto output = at::empty_like(contiguous_input);
            // SOFTMAX_MODE model=SOFTMAX_MODE::LOGSOFTMAX;
            SOFTMAX_MODE model=SOFTMAX_MODE::SOFTMAX;
            ::musa::dnn::Softmax softmax;
            at::impl::musa::muTensor musa_out = at::impl::musa::CreateMUTensor(_out);
            at::impl::musa::muTensor musa_self = at::impl::musa::CreateMUTensor(output);
            CHECK_MUDNN_STATUS(softmax.SetMode(model), "SetMode");
            CHECK_MUDNN_STATUS(softmax.SetDim(static_cast<int>(dim)), "SetDim");
            CHECK_MUDNN_STATUS(
                softmax.SetAlgorithm(::musa::dnn::Softmax::Algorithm::ACCURATE),
                "SetAlgorithm");
            CHECK_MUDNN_STATUS(softmax.Run(h, musa_out, musa_self), "Run");
            return diopiSuccess;
}

DIOPI_API diopiError_t diopiRsqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input){
            musa_torch::Tensor _out = at::impl::musa::build_musatorch_tensor(out);
            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            at::impl::musa::muTensor musa_self = at::impl::musa::CreateMUTensor(_input);
            at::impl::musa::muTensor musa_out = at::impl::musa::CreateMUTensor(_out);
            at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();
            UNARY_MODE m = UNARY_MODE::RSQRT;
            ::musa::dnn::Unary bop;
            bop.SetMode(m);
            bop.Run(h, musa_out, musa_self);
            std::cout << "=================hello,diopiRsqrt========================\n";
            return diopiSuccess;
}

DIOPI_API diopiError_t diopiSin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input){
            musa_torch::Tensor _out = at::impl::musa::build_musatorch_tensor(out);
            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            at::impl::musa::muTensor musa_self = at::impl::musa::CreateMUTensor(_input);
            at::impl::musa::muTensor musa_out = at::impl::musa::CreateMUTensor(_out);
            at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();
            UNARY_MODE m = UNARY_MODE::SIN;
            ::musa::dnn::Unary bop;
            bop.SetMode(m);
            bop.Run(h, musa_out, musa_self);
            std::cout << "=================hello,Sin========================\n";
            return diopiSuccess;
}
DIOPI_API diopiError_t diopiCos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input){
            musa_torch::Tensor _out = at::impl::musa::build_musatorch_tensor(out);
            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            at::impl::musa::muTensor musa_self = at::impl::musa::CreateMUTensor(_input);
            at::impl::musa::muTensor musa_out = at::impl::musa::CreateMUTensor(_out);
            at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();
            UNARY_MODE m = UNARY_MODE::COS;
            ::musa::dnn::Unary bop;
            bop.SetMode(m);
            bop.Run(h, musa_out, musa_self);
            std::cout << "=================hello,COS========================\n";
            return diopiSuccess;
}
DIOPI_API diopiError_t diopiSilu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input){
            musa_torch::Tensor _out = at::impl::musa::build_musatorch_tensor(out);
            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            at::impl::musa::muTensor musa_self = at::impl::musa::CreateMUTensor(_input);
            at::impl::musa::muTensor musa_out = at::impl::musa::CreateMUTensor(_out);
            at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();
            UNARY_MODE m = UNARY_MODE::SILU;
            ::musa::dnn::Unary bop;
            bop.SetMode(m);
            bop.Run(h, musa_out, musa_self);
            std::cout << "=================hello,diopiSilu========================\n";
            return diopiSuccess;
}

DIOPI_API diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value){
            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            musa_torch::Scalar _other = at::impl::musa::build_musatorch_scalar(value);
            at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();
            at::impl::musa::muTensor musa_self = at::impl::musa::CreateMUTensor(_input);
            ::musa::dnn::Fill op;
            auto other_scalar = _other;
            if (_input.scalar_type() == c10::kLong) {
            // if (other_scalar.isFloatingPoint()) {
            std::cout << "=================== c10::kLong==========================\n";
            CHECK_MUDNN_STATUS(op.SetValue(other_scalar.toLong()), "SetValue");
            } 
            else if (_input.scalar_type() == c10::kFloat) { 
            std::cout << "=================== c10::kFloat ==========================\n";
            CHECK_MUDNN_STATUS(op.SetValue(other_scalar.toFloat()), "SetValue"); 
            }
            else {
            std::cout << "=================== not is c10::kLong==========================\n";
            CHECK_MUDNN_STATUS(op.SetValue(other_scalar.toDouble()), "SetValue");
            }
            // CHECK_MUDNN_STATUS(op.SetValue(0.0), "SetValue");
            CHECK_MUDNN_STATUS(op.Run(h, musa_self), "Run");
            std::cout << "=================hello,diopifill========================\n";
            return diopiSuccess;
}
// 整数不支持
DIOPI_API diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* exponent){
            musa_torch::Tensor _out = at::impl::musa::build_musatorch_tensor(out);
            std::cout << "---------The data type of _out is---------: " << toString(_out.scalar_type()) << std::endl;
            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            musa_torch::Scalar _other = at::impl::musa::build_musatorch_scalar(exponent);        
            bool is_other_integer = false;
            at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();
            at::impl::musa::muTensor musa_self = at::impl::musa::CreateMUTensor(_input);
            at::impl::musa::muTensor musa_out = at::impl::musa::CreateMUTensor(_out);     
            ::musa::dnn::Unary uop;
            auto other_scalar = _other;
            if (other_scalar.isFloatingPoint()) {
                CHECK_MUDNN_STATUS(uop.SetAlpha(other_scalar.toDouble()), "SetAlpha");
                std::cout << "================isFloatingPoint=======================\n";         
            } else if (other_scalar.isIntegral(false)) {
                std::cout << "================isIntegral=======================\n";         
                CHECK_MUDNN_STATUS(uop.SetAlpha(other_scalar.toLong()), "SetAlpha");
            } 
            else {
                AT_ERROR(
                    other_scalar.type(), " is not implemented for broadcast in Binary");
            }
            CHECK_MUDNN_STATUS(uop.SetMode(UNARY_MODE::POW), "SetMode");
            CHECK_MUDNN_STATUS(uop.Run(h, musa_out, musa_self),"Run diopiPow");
            std::cout << "=================hello,diopiPow========================\n";
            return diopiSuccess;
}
// greater than or equal
DIOPI_API diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other){
            musa_torch::Tensor _out = at::impl::musa::build_musatorch_tensor(out);
            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            musa_torch::Tensor _other = at::impl::musa::build_musatorch_tensor(other);
            BINARY_MODE m = BINARY_MODE::GE;
            at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();
            at::impl::musa::muTensor musa_self = at::impl::musa::CreateMUTensor(_input);
            at::impl::musa::muTensor musa_out = at::impl::musa::CreateMUTensor(_out);
            at::impl::musa::muTensor musa_other = at::impl::musa::CreateMUTensor(_other);
            ::musa::dnn::Binary bop;
            bop.SetMode(m);
            bop.Run(h, musa_out, musa_self, musa_other);
            std::cout << "=================hello,diopiGe========================\n";
            return diopiSuccess;
}

DIOPI_API diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other){
            musa_torch::Tensor _out = at::impl::musa::build_musatorch_tensor(out);
            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            musa_torch::Tensor _other = at::impl::musa::build_musatorch_tensor(other);
            BINARY_MODE m = BINARY_MODE::EQ;
            at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();
            at::impl::musa::muTensor musa_self = at::impl::musa::CreateMUTensor(_input);
            at::impl::musa::muTensor musa_out = at::impl::musa::CreateMUTensor(_out);
            at::impl::musa::muTensor musa_other = at::impl::musa::CreateMUTensor(_other);
            ::musa::dnn::Binary bop;
            bop.SetMode(m);
            bop.Run(h, musa_out, musa_self, musa_other);
            std::cout << "=================hello,diopiEQ01========================\n";
            return diopiSuccess;
}

DIOPI_API diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other){
            musa_torch::Tensor _out = at::impl::musa::build_musatorch_tensor(out);
            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            musa_torch::Tensor _other = at::impl::musa::build_musatorch_tensor(other);
            BINARY_MODE m = BINARY_MODE::NE;
            at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();
            at::impl::musa::muTensor musa_self = at::impl::musa::CreateMUTensor(_input);
            at::impl::musa::muTensor musa_out = at::impl::musa::CreateMUTensor(_out);
            at::impl::musa::muTensor musa_other = at::impl::musa::CreateMUTensor(_other);
            ::musa::dnn::Binary bop;
            bop.SetMode(m);
            bop.Run(h, musa_out, musa_self, musa_other);
            std::cout << "=================hello,diopiNE========================\n";
            return diopiSuccess;
}

DIOPI_API diopiError_t diopiGt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other){
            musa_torch::Tensor _out = at::impl::musa::build_musatorch_tensor(out);
            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            musa_torch::Tensor _other = at::impl::musa::build_musatorch_tensor(other);
            at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();
            at::impl::musa::muTensor musa_self = at::impl::musa::CreateMUTensor(_input);
            at::impl::musa::muTensor musa_out = at::impl::musa::CreateMUTensor(_out);
            at::impl::musa::muTensor musa_other = at::impl::musa::CreateMUTensor(_other);
            BINARY_MODE m = BINARY_MODE::GT;
            ::musa::dnn::Binary bop;
            bop.SetMode(m);
            bop.Run(h, musa_out, musa_self, musa_other);
            std::cout << "=================hello,diopiGT========================\n";
            return diopiSuccess;
}

// // less than 没问题
DIOPI_API diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other){
            musa_torch::Tensor _out = at::impl::musa::build_musatorch_tensor(out);
            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            musa_torch::Tensor _other = at::impl::musa::build_musatorch_tensor(other);
            at::impl::musa::muHandle& h = at::impl::musa::GetMudnnHandle();
            at::impl::musa::muTensor musa_self = at::impl::musa::CreateMUTensor(_input);
            at::impl::musa::muTensor musa_out = at::impl::musa::CreateMUTensor(_out);
            at::impl::musa::muTensor musa_other = at::impl::musa::CreateMUTensor(_other);
            BINARY_MODE m = BINARY_MODE::LT;
            ::musa::dnn::Binary bop;
            bop.SetMode(m);
            bop.Run(h, musa_out, musa_self, musa_other);
            std::cout << "=================hello,diopiLT========================\n";
            return diopiSuccess;
}


DIOPI_API diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2){
            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            musa_torch::Tensor mat2_ = at::impl::musa::build_musatorch_tensor(mat2);
            musa_torch::Tensor out_ = at::impl::musa::build_musatorch_tensor(out);

            // musa_torch::Tensor result = at::empty(
            //     {_input.size(0), mat2_.size(1)},
            //                     // _input.options().memory_format(at::MemoryFormat::Contiguous)
            //                        _input.options().memory_format(at::MemoryFormat::Contiguous).device(at::kPrivateUse1)
            //    );              
            at::impl::musa::MmOut(_input, mat2_, out_);
            std::cout << "=================hello,diopiMm========================\n";
            return diopiSuccess;  

}


DIOPI_API diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2){
            musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
            musa_torch::Tensor mat2_ = at::impl::musa::build_musatorch_tensor(mat2);
            musa_torch::Tensor out_ = at::impl::musa::build_musatorch_tensor(out);
            std::cout << "================diopiBmm::out_::scalar_type======="<<out_.scalar_type()<<"=================\n";         
            std::cout << "================diopiBmm::out_::device======="<<out_.device()<<"=================\n";   
            PrintTensorValues(out_);

            // out_ = at::empty(
            // // musa_torch::Tensor result = at::empty(
            //     {_input.size(0), _input.size(1), mat2_.size(2)},
            //         // _input.options().memory_format(at::MemoryFormat::Contiguous)
            //        _input.options().memory_format(at::MemoryFormat::Contiguous).device(at::kPrivateUse1)
            //        );

            out_=at::impl::musa::BmmOut(_input, mat2_, out_);

            std::cout << "=================hello,diopiBmm========================\n";
            return diopiSuccess;  
}


// DIOPI_API diopiError_t diopiMaxAll(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t max_indices, diopiConstTensorHandle_t input,
//                                 int64_t dim){

// DIOPI_API diopiError_t diopiMaxAll(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiConstTensorHandle_t input){
    
//                  std::cout << "=================hello,diopiMaxAll========================\n";
//                  musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
//                  musa_torch::Tensor _max = at::impl::musa::build_musatorch_tensor(max);
//                 //  musa_torch::Tensor _max_indices = at::impl::musa::build_musatorch_tensor(max_indices);                    
//                 //  if (_input.scalar_type() == ScalarType::Double) {
//                 //     return at::max(_input.to("cpu")).to("dipu");
//                 //  }
//                 REDUCE_MODE m = REDUCE_MODE::MAX;           

//                 c10::DimVector dims_vec(0);
//                 if (_input.numel() == 0) {
//                     _max.zero_();
//                 } else {
//                     at::impl::musa::ReduceCall(_max, _input, dims_vec, m);
//                 }
//  }

DIOPI_API diopiError_t diopiMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t max_indices, diopiConstTensorHandle_t input,
                                int64_t dim){
                 musa_torch::Tensor _input = at::impl::musa::build_musatorch_tensor(input);
                 musa_torch::Tensor _max = at::impl::musa::build_musatorch_tensor(max);
                //  musa_torch::Tensor _max_indices = at::impl::musa::build_musatorch_tensor(max_indices);                    
                //  if (_input.scalar_type() == ScalarType::Double) {
                //     return at::max(_input.to("cpu")).to("dipu");
                //  }
                REDUCE_MODE m = REDUCE_MODE::MAX;           
                c10::DimVector dims_vec(0);
                if (_input.numel() == 0) {
                    _max.zero_();
                } else {
                    at::impl::musa::ReduceCall(_max, _input, dims_vec, m);
                }
                 std::cout << "=================hello,diopiMax========================\n";

 }













}   //extern C
}  // namespace musa
}  // namespace impl
