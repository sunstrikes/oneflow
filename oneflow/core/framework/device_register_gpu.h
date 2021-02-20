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

#include <type_traits>
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <cudnn.h>
#include <nccl.h>
#include <cuda_fp16.h>
#endif  // WITH_CUDA
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/device_registry_manager.h"

namespace oneflow {
#ifdef WITH_CUDA
void GpuDumpVersionInfo();

template<typename T>
struct IsFloat16;

template<>
struct IsFloat16<half> : std::true_type {};

REGISTER_DEVICE(DeviceType::kGPU).SetDumpVersionInfoFn(GpuDumpVersionInfo).SetDeviceTag("gpu");
#endif  // WITH_CUDA
}  // namespace oneflow
