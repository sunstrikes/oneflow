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
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/instructions_builder.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<InstructionsBuilder, std::shared_ptr<InstructionsBuilder>>(m, "InstructionsBuilder")
      .def(py::init([](const std::shared_ptr<vm::IdGenerator>& id_generator,
                       const std::shared_ptr<vm::cfg::InstructionListProto>& instruction_list,
                       const std::shared_ptr<eager::cfg::EagerSymbolList>& symbol_list,
                       const std::function<void(compatible_py::BlobObject*)>& release_object) {
        return std::make_shared<InstructionsBuilder>(id_generator, instruction_list, symbol_list,
                                                     release_object);
      }))
      .def("id_generator", &InstructionsBuilder::id_generator)
      .def("instruction_list", &InstructionsBuilder::instruction_list)
      .def("eager_symbol_list", &InstructionsBuilder::eager_symbol_list)
      .def("release_object", &InstructionsBuilder::release_object)
      .def("_NewSymbolId", &InstructionsBuilder::NewSymbolId)
      .def("_NewObjectId", &InstructionsBuilder::NewObjectId)
      .def("_NewSymbolId4String", &InstructionsBuilder::NewSymbolId4String)
      .def("GetSymbol4String", &InstructionsBuilder::GetSymbol4String)
      .def("GetJobConfSymbol", &InstructionsBuilder::GetJobConfSymbol)
      .def("GetParallelDescSymbol", &InstructionsBuilder::GetParallelDescSymbol)
      .def("GetScopeSymbol", &InstructionsBuilder::GetScopeSymbol)
      .def("NewSymbolId4OpNodeSignature", &InstructionsBuilder::NewSymbolId4OpNodeSignature)
      .def("BroadcastBlobReference", &InstructionsBuilder::BroadcastBlobReference)
      .def("GetPhysicalParallelDescSymbols", &InstructionsBuilder::GetPhysicalParallelDescSymbols)
      .def("BuildScopeByProtoSetter", &InstructionsBuilder::BuildScopeByProtoSetter)
      .def("BuildScopeWithNewIsMirrored", &InstructionsBuilder::BuildScopeWithNewIsMirrored)
      .def("BuildScopeWithNewScopeName", &InstructionsBuilder::BuildScopeWithNewScopeName)
      .def("BuildSendInstruction", &InstructionsBuilder::BuildSendInstruction)
      .def("BuildRecvInstruction", &InstructionsBuilder::BuildRecvInstruction)
      .def("CudaHostRegisterBlob", &InstructionsBuilder::CudaHostRegisterBlob)
      .def("CudaHostUnregisterBlob", &InstructionsBuilder::CudaHostUnregisterBlob)
      .def("_NewBlobObject", &InstructionsBuilder::NewBlobObject)
      .def("NewSharedOpKernelObjectId4ParallelConfSymbolId",
           &InstructionsBuilder::NewSharedOpKernelObjectId4ParallelConfSymbolId)
      .def("LazyReference", &InstructionsBuilder::LazyReference)
      .def("ReplaceMirrored", &InstructionsBuilder::ReplaceMirrored)
      .def("DeleteObject", &InstructionsBuilder::DeleteObject);
}

}  // namespace oneflow