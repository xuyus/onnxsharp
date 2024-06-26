# Copyright 2024 XUYUS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import os
import numpy
from onnx import TensorProto, numpy_helper
from .npy_utils import npy_summurize_array


def ort_scan_tensor_from_dump(dump_dir, tensor_name):
    filename = os.path.join(dump_dir, f"{tensor_name}.tensorproto")
    if not os.path.exists(filename):
        return False

    with open(filename, "rb") as f:
        tensor = TensorProto()
        tensor.ParseFromString(f.read())
        array = numpy_helper.to_array(tensor)
        flatten_array = array.flatten()
        return npy_summurize_array(flatten_array, name=tensor_name)


def ort_get_tensor_from_dump(dump_dir, tensor_name):
    filename = os.path.join(dump_dir, f"{tensor_name}.tensorproto")
    if not os.path.exists(filename):
        raise RuntimeError(f"tensor {tensor_name} not found in {dump_dir}")
        return None

    with open(filename, "rb") as f:
        tensor = TensorProto()
        tensor.ParseFromString(f.read())
        array = numpy_helper.to_array(tensor)
        flatten_array = array.flatten()

        return flatten_array
