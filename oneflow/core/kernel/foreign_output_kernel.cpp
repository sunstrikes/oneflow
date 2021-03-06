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
#include "oneflow/core/kernel/foreign_output_kernel.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/job/foreign_job_instance.h"

namespace oneflow {

void ForeignOutputKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& buffer_name = op_conf().foreign_output_conf().ofblob_buffer_name();
  std::shared_ptr<ForeignJobInstance> foreign_job_instance;
  BufferStatus buffer_status = Global<BufferMgr<std::shared_ptr<ForeignJobInstance>>>::Get()
                                   ->Get(buffer_name)
                                   ->TryReceive(&foreign_job_instance);
  CHECK_NE(buffer_status, kBufferStatusEmpty);
  if (buffer_status == kBufferStatusSuccess) {
    OfBlob ofblob(ctx.device_ctx, BnInOp2Blob("in"));
    foreign_job_instance->PullBlob(reinterpret_cast<uint64_t>(&ofblob));
  }
}

REGISTER_KERNEL(OperatorConf::kForeignOutputConf, ForeignOutputKernel);

}  // namespace oneflow
