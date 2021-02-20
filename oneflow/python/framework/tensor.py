"""
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
"""
import oneflow_api
import oneflow.python.framework.dtype as dtype_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.device as oneflow_device


def get_default_dtype():
    return dtype_util.float


def is_consistent_now():
    return False


@oneflow_export("Tensor")
class Tensor:
    def __init__(
        self, shape: tuple = (), device: oneflow_device = oneflow_device.Device("cpu"),
    ):
        self._dtype = get_default_dtype()
        self._shape = shape
        self._device = device
        # a MirroredTensor or ConsistentTensor object
        self._impl = None

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def _initialize(self):
        if is_consistent_now():
            raise NotImplementedError()
        else:
            self._impl = MirroredTensor(self.shape, self.dtype, self.device)

    def _initialize_with_data(self, data):
        raise NotImplementedError()


@oneflow_export("MirroredTensor")
class MirroredTensor(oneflow_api.MirroredTensor):
    def __init__(
        self,
        shape: tuple = (),
        dtype: dtype_util.dtype = dtype_util.float,
        device: oneflow_device = oneflow_device.Device("cpu"),
    ):
        of_dtype = dtype_util.convert_oneflow_dtype_to_proto_dtype(dtype)
        oneflow_api.MirroredTensor.__init__(self, shape, of_dtype, device)

    @property
    def dtype(self):
        return dtype_util.convert_proto_dtype_to_oneflow_dtype(self.get_dtype())
