from collections import OrderedDict
from black import validate_cell
import onnx
from onnx import helper, defs, numpy_helper, checker
import copy
from .basics import enforce, Type


class TensorShape(object):
    def __init__(self, tensor_shape_proto) -> None:
        self._dim = []
        for dim_proto in tensor_shape_proto.dim:
            if dim_proto.HasField("dim_param"):
                self._dim.append(dim_proto.dim_param)
            elif dim_proto.HasField("dim_value"):
                self._dim.append(dim_proto.dim_value)

    def to_proto(self):
        tensor_shape_proto = onnx.TensorShapeProto()
        dim_protos = []
        for d in self._dim:
            if isinstance(d, str):
                d_proto = onnx.TensorShapeProto.Dimension()
                d_proto.dim_param = str(d)
                dim_protos.append(d_proto)

        tensor_shape_proto.dim.extend(dim_protos)
        return tensor_shape_proto


class TensorType(Type):
    def __init__(self, tensor_type_proto) -> None:
        # int32 elem_type = 1;
        self._elem_type = tensor_type_proto.elem_type

        # TensorShapeProto shape = 2;
        self._shape = TensorShape(tensor_type_proto.shape)

    def to_proto(self):
        type_proto = onnx.TypeProto()
        tensor_type_proto = type_proto.tensor_type
        tensor_type_proto.elem_type = self._elem_type
        tensor_type_proto.shape.CopyFrom(self._shape.to_proto())
        return type_proto


class Tensor(object):
    """// A serialized tensor value."""

    def __init__(self, tensor_proto) -> None:
        self._tensor_proto = tensor_proto

    def to_proto(self):
        return self._tensor_proto
