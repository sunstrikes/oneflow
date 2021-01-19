/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef _ONEFLOW_USER_KERNELS_ELEMENTWISE_HARDSWISH_KERNEL_H_
#define _ONEFLOW_USER_KERNELS_ELEMENTWISE_HARDSWISH_KERNEL_H_
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

template<typename T>
struct HardswishFunctor {
  OF_DEVICE_FUNC T operator()(const T x) const {
    if (x <= static_cast<T>(-3)) {
      return static_cast<T>(0);
    } else if (x >= static_cast<T>(3)) {
      return x;
    } else {
      return (x * (x + static_cast<T>(3))) / static_cast<T>(6);
    }
  }
};

template<typename T>
struct HardswishGradFunctor {
  OF_DEVICE_FUNC T operator()(const T x, const T dy) const {
    if (x <= static_cast<T>(-3)) {
      return static_cast<T>(0);
    } else if (x >= static_cast<T>(3)) {
      return dy;
    } else {
      return ((x / static_cast<T>(3)) + static_cast<T>(0.5)) * dy;
    }
  }
};

namespace {

template<DeviceType device_type, template<typename> class Opt, typename T>
struct ElemwiseHardswishFunctor final {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const T* in, T* out);
};

template<DeviceType device_type, template<typename> class Opt, typename T>
struct ElemwiseHardswishGradFunctor final {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const T* x, const T* dy, T* dx);
};

}  // namespace

template<DeviceType device_type, template<typename> class Opt, typename T>
class ElemwiseHardswishKernel final : public user_op::OpKernel {
 public:
  ElemwiseHardswishKernel() = default;
  ~ElemwiseHardswishKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const T* in_ptr = in_tensor->dptr<T>();
    T* out_ptr = out_tensor->mut_dptr<T>();
    const int64_t elem_cnt = in_tensor->shape().elem_cnt();

    ElemwiseHardswishFunctor<device_type, Opt, T>()(ctx->device_ctx(), elem_cnt, out_ptr, in_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, template<typename> class Opt, typename T>
class ElemwiseHardswishGradKernel final : public user_op::OpKernel {
 public:
  ElemwiseHardswishGradKernel() = default;
  ~ElemwiseHardswishGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const T* x_ptr = x_tensor->dptr<T>();
    const T* dy_ptr = dy_tensor->dptr<T>();
    T* dx_ptr = dx_tensor->mut_dptr<T>();

    const int64_t elem_cnt = x_tensor->shape().elem_cnt();

    ElemwiseHardswishGradFunctor<device_type, Opt, T>()(ctx->device_ctx(), elem_cnt, dx_ptr, x_ptr,
                                                        dy_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_HARDSWISH_KERNELS(device, dtype)                                       \
  REGISTER_USER_KERNEL("hardswish")                                                     \
      .SetCreateFn<ElemwiseHardswishKernel<device, HardswishFunctor, dtype>>()          \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("hardswish_grad")                                                \
      .SetCreateFn<ElemwiseHardswishGradKernel<device, HardswishGradFunctor, dtype>>()  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

}  // namespace oneflow

#endif  // _ONEFLOW_USER_KERNELS_ELEMENTWISE_HARDSWISH_KERNEL_H_