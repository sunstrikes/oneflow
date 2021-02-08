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
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/api/python/job_build/job_build_and_infer.h"

namespace oneflow {
namespace one {

void OpExprInterpreter::ResetSelfState() { self_state_.reset(new OpExprInterpState); }

void NormalInterpreter::Apply(const OpExpr* op_expr, const TensorList& inputs, TensorList& outputs,
                              const OpExprInterpState* state) {
  ResetSelfState();
  if (op_expr->type() == "UserOp") {
    Apply_(dynamic_cast<const UserOpExpr*>(op_expr), inputs, outputs, state);
  } else if (op_expr->type() == "FunctionOp") {
    Apply_(dynamic_cast<const FunctionOpExpr*>(op_expr), inputs, outputs, state);
  } else {
    LOG(FATAL) << "The op type " << op_expr->type()
               << " is not supported in LazyInterpreter::Apply currently.";
  }
}

OpAttribute AddOpAndInferAttribute(const OpExprInterpContext* ctx, OperatorConf& op_conf) {
  int64_t symbol_id = ctx->scope->symbol_id().GetOrThrow();
  op_conf.set_scope_symbol_id(symbol_id);
  if (!op_conf.has_device_tag()) {
    op_conf.set_device_tag(ctx->scope->device_parallel_desc_symbol()->device_tag());
  }

  auto infer_ctx = GetCurInferCtx().GetOrThrow();
  if (ctx->is_mirrored_strategy_enabled) {
    return infer_ctx->AddAndInferMirroredOp(op_conf).GetOrThrow();
  } else {
    return infer_ctx->AddAndInferConsistentOp(op_conf).GetOrThrow();
  }
}

void LazyInterpreter::Apply_(const UserOpExpr* op_expr, const TensorList& inputs,
                             TensorList& outputs, const OpExprInterpState* state) {
  OperatorConf op_conf;
  *(op_conf.mutable_user_conf()) = op_expr->proto();
  *(op_conf.mutable_name()) = op_expr->op_name();

  auto op_attribute = AddOpAndInferAttribute(context_, op_conf);

  // Check outputs num and setup output tensors properties.
  CHECK_EQ(outputs.size(), op_expr->output_num());
  int i = 0;
  for (const auto& it : op_expr->proto().output()) {
    for (const auto& output_name : it.second.s()) {
      // TODO
      // outputs[i];
      TensorNameScope::Global()->Record(outputs[i], output_name);
    }
  }
}

void LazyInterpreter::Apply_(const FunctionOpExpr* op_expr, const TensorList& inputs,
                             TensorList& outputs, const OpExprInterpState* state) {
  // TODO(hjchen2)
}

void EagerInterpreter::Apply_(const UserOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  OperatorConf op_conf;
  *(op_conf.mutable_user_conf()) = op_expr->proto();
  *(op_conf.mutable_name()) = op_expr->op_name();

  auto op_attribute = AddOpAndInferAttribute(context_, op_conf);
  const auto& parallel_conf = context_->scope->device_parallel_desc_symbol()->parallel_conf();

  auto cfg_op_attribute = std::make_shared<cfg::OpAttribute>(op_attribute);
  auto cfg_parallel_conf = std::make_shared<cfg::ParallelConf>(parallel_conf);
  auto BuildInstruction = [cfg_op_attribute,
                           cfg_parallel_conf](const std::shared_ptr<InstructionsBuilder>& builder) {
    // TODO(hjchen2) Complete bn2blob_object and find_or_creat_blob_object_fn.
    std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>
        bn2blob_object;
    InstructionsBuilder::FindOrCreateDelegateBlobObjectFun find_or_creat_blob_object_fn;
    builder->NoBoxingStatelessCall(cfg_op_attribute, cfg_parallel_conf, bn2blob_object,
                                   find_or_creat_blob_object_fn);
  };
  void(LogicalRun(BuildInstruction).GetOrThrow());
}

void EagerInterpreter::Apply_(const FunctionOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  // TODO(hjchen2)
}

void AutogradInterpreter::Apply(const OpExpr* op_expr, const TensorList& inputs,
                                TensorList& outputs, const OpExprInterpState* state) {
  // TODO(hjchen2)
}

}  // namespace one
}  // namespace oneflow
