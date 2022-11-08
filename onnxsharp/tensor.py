from collections import OrderedDict
from black import validate_cell
import onnx
from onnx import helper, defs, numpy_helper, checker, onnx_pb
import copy
from .basics import enforce, Type
from onnx.numpy_helper import to_array, from_array
import numpy as np

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

        self._raw_value = None
        self._float_data = None
        self._int32_data = None
        self._int64_data = None
        self._uint64_data = None
        self._double_data = None
        self._string_data = None

    @classmethod
    def from_proto(self, tensor_proto):
        t = Tensor()

        ## The shape of the tensor.
        # repeated int64 dims = 1;
        # if tensor_proto.HasField("dim"):
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
        t._name = tensor_proto.name if tensor_proto.HasField("name") else None

        # optional string doc_string = 12;

        # optional bytes raw_data = 9;

        # repeated StringStringEntryProto external_data = 13;

        # optional DataLocation data_location = 14;

        # repeated double double_data = 10 [packed = true];

        # repeated uint64 uint64_data = 11 [packed = true];

        if tensor_proto.HasField("raw_data"):
            # print("raw_data found for tensorproto.name", tensor_proto.name)
            t._raw_value = numpy_helper.to_array(tensor_proto)
            return t

        if tensor_proto.data_type in [
            onnx_pb.TensorProto.FLOAT,
            onnx_pb.TensorProto.COMPLEX64,
        ]:
            t._float_data = np.array([f for f in tensor_proto.float_data], dtype=np.float32)

        if tensor_proto.data_type in [
            onnx_pb.TensorProto.FLOAT16,
            onnx_pb.TensorProto.BFLOAT16,
            onnx_pb.TensorProto.BOOL,
            onnx_pb.TensorProto.INT8,
            onnx_pb.TensorProto.INT16,
            onnx_pb.TensorProto.INT32,
            onnx_pb.TensorProto.UINT8,
            onnx_pb.TensorProto.UINT16,
        ]:
            t._int32_data = np.array([i for i in tensor_proto.int32_data], dtype=np.int32)

        if tensor_proto.data_type in [
            onnx_pb.TensorProto.INT64,
        ]:
            t._int64_data = np.array([i for i in tensor_proto.int64_data], dtype=np.int64)

        if tensor_proto.data_type in [
            onnx_pb.TensorProto.UINT32,
            onnx_pb.TensorProto.UINT64,
        ]:
            t._uint64_data = np.array([i for i in tensor_proto.uint64_data], dtype=np.uint64)

        if tensor_proto.data_type in [
            onnx_pb.TensorProto.DOUBLE,
            onnx_pb.TensorProto.COMPLEX128,
        ]:
            t._double_data = np.array([i for i in tensor_proto.double_data], dtype=np.float64)

        if tensor_proto.data_type in [
            onnx_pb.TensorProto.STRING,
        ]:
            t._string_data = np.array([i for i in tensor_proto.string_data], dtype=np.object)

        return t

    @property
    def value(self):

        if self._float_data is not None:
            return self._float_data

        if self._int32_data is not None:
            return self._int32_data

        if self._int64_data is not None:
            return self._int64_data

        if self._uint64_data is not None:
            return self._uint64_data

        if self._double_data is not None:
            return self._double_data

        if self._string_data is not None:
            return self._string_data

        if self._raw_value is not None:
            return self._raw_value

    def to_proto(self):
        if self._raw_value is not None:
            return numpy_helper.from_array(self._raw_value, name=self._name)

        tensor_proto = onnx.TensorProto()
        if self._data_type in [
            onnx_pb.TensorProto.FLOAT,
            onnx_pb.TensorProto.COMPLEX64,
        ]:
            tensor_proto.float_data.extend(self._float_data.tolist())

        if self._data_type in [
            onnx_pb.TensorProto.FLOAT16,
            onnx_pb.TensorProto.BFLOAT16,
            onnx_pb.TensorProto.BOOL,
            onnx_pb.TensorProto.INT8,
            onnx_pb.TensorProto.INT16,
            onnx_pb.TensorProto.INT32,
            onnx_pb.TensorProto.UINT8,
            onnx_pb.TensorProto.UINT16,
        ]:
            tensor_proto.int32_data.extend(self._int32_data.tolist())

        if self._data_type in [
            onnx_pb.TensorProto.INT64,
        ]:
            tensor_proto.int64_data.extend(self._int64_data.tolist())

        if self._data_type in [
            onnx_pb.TensorProto.UINT32,
            onnx_pb.TensorProto.UINT64,
        ]:
            tensor_proto.uint64_data.extend(self._uint64_data.tolist())

        if self._data_type in [
            onnx_pb.TensorProto.DOUBLE,
            onnx_pb.TensorProto.COMPLEX128,
        ]:
            tensor_proto.double_data.extend(self._double_data.tolist())

        if self._data_type in [
            onnx_pb.TensorProto.STRING,
        ]:
            tensor_proto.string_data.extend(self._string_data.tolist())

        tensor_proto.dims.extend(self._dims)
        tensor_proto.data_type = self._data_type

        if self._name:
            tensor_proto.name = self._name

        return tensor_proto

    def __str__(self):
        return f"Tensor: name - {self._name}, type - {self._data_type}, dims - {self._dims}"

    def __repr__(self):
        return str(self)
