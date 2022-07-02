from collections import OrderedDict
from black import validate_cell
import onnx
from onnx import helper, defs, numpy_helper, checker, onnx_pb
import copy
from .basics import enforce, Type
from onnx.numpy_helper import to_array, from_array


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
            else:
                d_proto = onnx.TensorShapeProto.Dimension()
                d_proto.dim_value = d
                dim_protos.append(d_proto)

        tensor_shape_proto.dim.extend(dim_protos)
        return tensor_shape_proto

    def __str__(self) -> str:
        return f"TensorShape - dims: {self._dim}"

    def __repr__(self) -> str:
        return str(self)


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

    def __str__(self) -> str:
        return f"TensorType - elem_type: {self._elem_type}, shape: {self._shape}"

    def __repr__(self) -> str:
        return str(self)


class Tensor(object):
    """// A serialized tensor value."""

    def __init__(self):
        self._dims = None
        self._data_type = None
        self._name = None
        self._value = None

    @classmethod
    def from_proto(self, tensor_proto):
        t = Tensor()

        ## The shape of the tensor.
        # repeated int64 dims = 1;
        t._dims = [int(dim_value) for dim_value in tensor_proto.dims]

        # optional int32 data_type = 2;
        t._data_type = tensor_proto.data_type

        # onnx_pb.TensorProto.FLOAT: np.float32,
        # onnx_pb.TensorProto.FLOAT16: np.float16,
        # onnx_pb.TensorProto.DOUBLE: np.float64,
        # onnx_pb.TensorProto.INT32: np.int32,
        # onnx_pb.TensorProto.INT16: np.int16,
        # onnx_pb.TensorProto.INT8: np.int8,
        # onnx_pb.TensorProto.UINT8: np.uint8,
        # onnx_pb.TensorProto.UINT16: np.uint16,
        # onnx_pb.TensorProto.UINT32: np.uint32,
        # onnx_pb.TensorProto.UINT64: np.uint64,
        # onnx_pb.TensorProto.INT64: np.int64,
        # onnx_pb.TensorProto.UINT64: np.uint64,
        # onnx_pb.TensorProto.BOOL: np.bool,
        # onnx_pb.TensorProto.COMPLEX64: np.complex64,
        # onnx_pb.TensorProto.COMPLEX128: np.complex128,
        # onnx_pb.TensorProto.STRING: np.object,

        # repeated float float_data = 4 [packed = true];

        # repeated int32 int32_data = 5 [packed = true];

        # repeated bytes string_data = 6;

        # repeated int64 int64_data = 7 [packed = true];

        # optional string name = 8; // namespace Value
        t._name = tensor_proto.name

        # optional string doc_string = 12;

        # optional bytes raw_data = 9;

        # repeated StringStringEntryProto external_data = 13;

        # optional DataLocation data_location = 14;

        # repeated double double_data = 10 [packed = true];

        # repeated uint64 uint64_data = 11 [packed = true];

        t._value = numpy_helper.to_array(tensor_proto)

        return t

    def to_proto(self):
        return numpy_helper.from_array(self._value, name=self._name)

    def __str__(self):
        return f"Tensor: name - {self._name}, type - {self._data_type}, dims - {self._dims}"

    def __repr__(self):
        return str(self)
