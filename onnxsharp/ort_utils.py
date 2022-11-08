import torch
import os
import numpy
from onnx import TensorProto, numpy_helper

def ort_scan_tensor_from_dump(dump_dir, tensor_name):
    filename = os.path.join(dump_dir, f"{tensor_name}.tensorproto")
    if not os.path.exists(filename):
        return False

    with open(filename, "rb") as f:
        tensor = TensorProto()
        tensor.ParseFromString(f.read())
        array = numpy_helper.to_array(tensor)
        flatten_array = array.flatten()
        print(f"{tensor_name} shape: {array.shape} dtype: {array.dtype} \n"
                f"min: {flatten_array.min()} max: {flatten_array.max()}, mean: {flatten_array.mean()}, std: {flatten_array.std()} \n"
                f"nan: {numpy.isnan(flatten_array).sum()}, inf: {numpy.isinf(flatten_array).sum()}, \n"
                f"neg: {numpy.less(flatten_array, 0).sum()}, pos: {numpy.greater(flatten_array, 0).sum()}, zero: {numpy.equal(flatten_array, 0).sum()}, \n"
                f"numpy.nonzero(array): {numpy.nonzero(flatten_array)}, \n"
                f"norm: {numpy.linalg.norm(flatten_array)}, l2: {numpy.linalg.norm(flatten_array, ord=2)}, \n"
                f"histogram: {numpy.histogram(flatten_array, bins=10)}")

        return numpy.isnan(flatten_array).sum() > 0 or numpy.isinf(flatten_array).sum() > 0
