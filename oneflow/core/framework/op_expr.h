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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_EXPR_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_EXPR_H_

#include <functional>

#include "oneflow/core/framework/bn_accessor.h"
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {
namespace one {

class Tensor {};
using TensorList = std::vector<std::shared_ptr<Tensor>>;

#define DEFINE_DEFAULT_CONSTRUCTOR(class_type) \
  class_type() = default;                      \
  virtual ~class_type() = default;

class OpExpr {
 public:
  DEFINE_DEFAULT_CONSTRUCTOR(OpExpr);

  // TODO(): Uncomment.
  // virtual FilterInputTensorsUsedByBackward(const TensorList& inputs) = 0;
  // virtual FilterOutputTensorsUsedByBackward(const TensorList& outputs) = 0;

  virtual std::shared_ptr<OpExpr> GetBackwardOpExpr() const = 0;
  virtual std::string type() const = 0;
};

class BuiltinOpExpr : public OpExpr {
 public:
  DEFINE_DEFAULT_CONSTRUCTOR(BuiltinOpExpr);

  explicit BuiltinOpExpr(const std::string& op_name) : op_name_(op_name) {}
  BuiltinOpExpr(const std::string& op_name, const std::vector<std::string>& indexed_input_names,
                const std::vector<std::string>& indexed_output_names)
      : op_name_(op_name),
        indexed_input_names_(indexed_input_names),
        indexed_output_names_(indexed_output_names) {}

  const std::string& op_name() const { return op_name_; }

  int input_num() const { return indexed_input_names_.size(); }
  int output_num() const { return indexed_output_names_.size(); }

  const std::vector<std::string>& indexed_input_names() const { return indexed_input_names_; }
  const std::vector<std::string>& indexed_output_names() const { return indexed_output_names_; }

  virtual void BuildOpConf(OperatorConf* op_conf,
                           std::function<std::string(const std::string&)> mutator) const = 0;

 protected:
  std::string op_name_;
  // The indexed input operand names.
  std::vector<std::string> indexed_input_names_;
  // The indexed output operand names.
  std::vector<std::string> indexed_output_names_;
};

#define DEFINE_BUILTIN_OPEXPR_CLASS(_op_name, _op_conf)                                       \
  class _op_name##Expr : public BuiltinOpExpr {                                               \
   public:                                                                                    \
    _op_name##Expr() = default;                                                               \
    virtual ~_op_name##Expr() = default;                                                      \
    explicit _op_name##Expr(const std::string& op_name, _op_name##Conf&& proto,               \
                            const std::vector<std::string>& indexed_input_names,              \
                            const std::vector<std::string>& indexed_output_names)             \
        : BuiltinOpExpr(op_name, indexed_input_names, indexed_output_names), proto_(proto) {} \
                                                                                              \
    std::shared_ptr<OpExpr> GetBackwardOpExpr() const override;                               \
                                                                                              \
    std::string type() const override { return std::string(#_op_name); }                      \
                                                                                              \
    const _op_name##Conf& proto() const { return proto_; }                                    \
    _op_name##Conf* mutable_proto() { return &proto_; }                                       \
                                                                                              \
    void BuildOpConf(OperatorConf* op_conf,                                                   \
                     std::function<std::string(const std::string&)> mutator) const {          \
      *(op_conf->mutable_name()) = this->op_name_;                                            \
      *(op_conf->mutable_##_op_conf##_conf()) = proto_;                                       \
      InOutbnAccessor<_op_name##Conf> io_accessor(op_conf->mutable_##_op_conf##_conf());      \
      for (std::string * input : io_accessor.input()) { *input = mutator(*input); }           \
    }                                                                                         \
                                                                                              \
   private:                                                                                   \
    _op_name##Conf proto_;                                                                    \
  };

DEFINE_BUILTIN_OPEXPR_CLASS(UserOp, user);
DEFINE_BUILTIN_OPEXPR_CLASS(VariableOp, variable);
DEFINE_BUILTIN_OPEXPR_CLASS(CastToMirroredOp, cast_to_mirrored);
DEFINE_BUILTIN_OPEXPR_CLASS(CastFromMirroredOp, cast_from_mirrored);
DEFINE_BUILTIN_OPEXPR_CLASS(DistributeSplitOp, distribute_split);
DEFINE_BUILTIN_OPEXPR_CLASS(DistributeCloneOp, distribute_clone);
DEFINE_BUILTIN_OPEXPR_CLASS(DistributeConcatOp, distribute_concat);
DEFINE_BUILTIN_OPEXPR_CLASS(DistributeAddOp, distribute_add);

// TODO(): Finish the class definition of `FunctionOpExpr`.
class FunctionOpExpr : public OpExpr {
 public:
  DEFINE_DEFAULT_CONSTRUCTOR(FunctionOpExpr);

  std::shared_ptr<OpExpr> GetBackwardOpExpr() const override;

  std::string type() const override { return "FunctionOp"; }
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_EXPR_H_