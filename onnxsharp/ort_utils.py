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


