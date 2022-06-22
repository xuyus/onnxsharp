from collections import OrderedDict
from black import validate_cell
import onnx
from onnx import helper, defs, numpy_helper, checker
import copy

from .tensor import TensorType, enforce


class ValueInfo(object):
    def __init__(self) -> None:
        self._name = None
        self._type = None
        self._doc_string = None

    @classmethod
    def from_proto(cls, value_info_proto):
        v = ValueInfo()
        # string name = 1;
        v._name = value_info_proto.name

        # TypeProto type = 2;
        type_proto = value_info_proto.type
        if type_proto.HasField("tensor_type"):
            v._type = TensorType(type_proto.tensor_type)
        else:
            raise NotImplementedError("unsupported type")

        # string doc_string = 3;
        v._doc_string = value_info_proto.doc_string

        return v

    @property
    def name(self):
        return self._name

    def set_name(self, new_name):
        self._name = new_name

    def to_proto(self):
        value_info_proto = onnx.ValueInfoProto()
        value_info_proto.name = self._name
        if self._type is not None:
            value_info_proto.type.CopyFrom(self._type.to_proto())
        if self._doc_string is not None:
            value_info_proto.doc_string = self._doc_string

        return value_info_proto

    def __str__(self) -> str:
        return f"ValueInfo - name: {self._name}, tensor type: {self._type}"

    def __repr__(self) -> str:
        return str(self)


class AttributeType:
    UNDEFINED = 0
    FLOAT = 1
    INT = 2
    STRING = 3
    TENSOR = 4
    GRAPH = 5
    SPARSE_TENSOR = 11
    TYPE_PROTO = 13

    FLOATS = 6
    INTS = 7
    STRINGS = 8
    TENSORS = 9
    GRAPHS = 10
    SPARSE_TENSORS = 12
    TYPE_PROTOS = 14


class Attribute(AttributeType):
    def __init__(self):
        # string name = 1
        self._name = None

        # string doc_string = 13;
        self._doc_string = None

        # AttributeType type = 20;
        self._type = None

        self._value = None

    @classmethod
    def from_proto(cls, attribute_proto):
        a = Attribute()
        a._name = attribute_proto.name
        a._doc_string = attribute_proto.doc_string
        a._type = attribute_proto.type

        if a._type == AttributeType.FLOAT:
            a._value = attribute_proto.f
        elif a._type == AttributeType.INT:
            a._value = attribute_proto.i
        elif a._type == AttributeType.STRING:
            a._value = attribute_proto.s
        elif a._type == AttributeType.TENSOR:
            a._value = attribute_proto.t
        elif a._type == AttributeType.GRAPH:
            a._value = attribute_proto.g
        elif a._type == AttributeType.SPARSE_TENSOR:
            a._value = attribute_proto.sparse_tensor
        elif a._type == AttributeType.TYPE_PROTO:
            a._value = attribute_proto.tp
        elif a._type == AttributeType.FLOATS:
            a._value = attribute_proto.floats
        elif a._type == AttributeType.INTS:
            a._value = attribute_proto.ints
        elif a._type == AttributeType.STRINGS:
            a._value = attribute_proto.strings
        elif a._type == AttributeType.TENSORS:
            a._value = attribute_proto.tensors
        elif a._type == AttributeType.GRAPHS:
            a._value = attribute_proto.graphs
        elif a._type == AttributeType.SPARSE_TENSORS:
            a._value = attribute_proto.sparse_tensors
        elif a._type == AttributeType.TYPE_PROTO:
            a._value = attribute_proto.type_protos

        return a

    def to_proto(self):
        attr_proto = onnx.AttributeProto()
        if self._type == AttributeType.FLOAT:
            attr_proto.f = self._value
        elif self._type == AttributeType.INT:
            attr_proto.i = self._value
        elif self._type == AttributeType.STRING:
            attr_proto.s = self._value
        elif self._type == AttributeType.TENSOR:
            attr_proto.t = self._value.to_proto()
        elif self._type == AttributeType.GRAPH:
            attr_proto.g = self._value.to_proto()
        elif self._type == AttributeType.SPARSE_TENSOR:
            attr_proto.sparse_tensor = self._value.to_proto()
        elif self._type == AttributeType.TYPE_PROTO:
            attr_proto.tp = self._value.to_proto()
        elif self._type == AttributeType.FLOATS:
            attr_proto.floats.extend(v for v in self._value)
        elif self._type == AttributeType.INTS:
            attr_proto.ints.extend(v for v in self._value)
        elif self._type == AttributeType.STRINGS:
            attr_proto.strings.extend(v for v in self._value)
        elif self._type == AttributeType.TENSORS:
            attr_proto.tensors.extend(v.to_proto() for v in self._value)
        elif self._type == AttributeType.GRAPHS:
            attr_proto.graphs.extend(v.to_proto() for v in self._value)
        elif self._type == AttributeType.SPARSE_TENSORS:
            attr_proto.sparse_tensors.extend(v.to_proto() for v in self._value)
        elif self._type == AttributeType.TYPE_PROTO:
            attr_proto.type_protos.extend(v.to_proto() for v in self._value)


class NodeArg(object):
    def __init__(self, name) -> None:
        self._name = name

        v = ValueInfo()
        # string name = 1;
        v._name = name
        self._value_info: ValueInfo = v

    @classmethod
    def from_valueinfo(cls, value_info: ValueInfo):
        instance = cls(value_info.name)
        instance._value_info = value_info
        return instance

    @property
    def name(self):
        return self._name

    def update(self, value_info):
        self._value_info = value_info

    def to_proto(self):
        return self._value_info.to_proto()

    def __str__(self) -> str:
        return f"NodeArg - name: {self._name}, value_info: {self._value_info}"

    def __repr__(self) -> str:
        return str(self)


class Node(object):
    def __init__(self) -> None:
        self._g = None

        ## ONNX Proto

        # self._node_proto = None

        # string name = 3;
        self._name = None

        # string op_type = 4;
        self._type = None

        # repeated string input = 1;
        self._input_args: list[NodeArg] = []

        # repeated string output = 2;
        self._output_args: list[NodeArg] = []

        # string domain = 7;
        self._domain = None

        # repeated AttributeProto attribute = 5;
        self._attr = OrderedDict()

        # string doc_string = 6;
        self._doc_string = None

        ## Execution Plan

        self._program_counter = None

    @classmethod
    def from_proto(self, graph, node_proto):
        n = Node()

        n._g = graph
        # n._node_proto = node_proto
        n._name = node_proto.name if node_proto.HasField("name") else None
        n._type = node_proto.op_type

        # Be noted, value of node_proto.input/output could be empty string
        # in ONNX, this means, the op have the input, but did not expect it connected to any
        # other nodes .
        n._input_args = [NodeArg(i) for i in node_proto.input]
        n._output_args = [NodeArg(o) for o in node_proto.output]

        n._domain = node_proto.domain if node_proto.HasField("domain") else None

        for a in node_proto.attribute:
            n._attr[a.name] = Attribute.from_proto(a)

        n._doc_string = (
            node_proto.doc_string if node_proto.HasField("doc_string") else None
        )

        return n

    def replace_input_arg(self, input_arg_name, new_arg: NodeArg):
        enforce(new_arg is not None, "new_arg could not be None")
        enforce(
            input_arg_name in self.input_arg_names,
            f"name: {input_arg_name} not exists",
        )

        index = self.input_arg_names.index(input_arg_name)
        self._input_args[index] = new_arg

    def replace_output_arg(self, output_arg_name, new_arg: NodeArg):
        enforce(new_arg is not None, "new_arg could not be None")
        enforce(
            output_arg_name in self.output_arg_names,
            f"name: {output_arg_name} not exists",
        )

        index = self.output_arg_names.index(output_arg_name)
        self._output_args[index] = new_arg

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def output_arg_names(self) -> list[str]:
        return [arg.name for arg in self._output_args]

    def output_arg(self, index):
        enforce(
            index is not None and index >= 0 and index < len(self._output_args),
            "index out of range",
        )
        return self._output_args[index]

    @property
    def input_arg_names(self) -> list[str]:
        return [arg.name for arg in self._input_args]

    def input_arg(self, index):
        enforce(
            index is not None and index >= 0 and index < len(self._input_args),
            "index out of range",
        )
        return self._input_args[index]

    def to_proto(self):
        attribute_protos = {
            attr_name: a.to_proto()
            for attr_name, a in self._attr.items()
            if attr_name != "name"
        }
        node_proto = helper.make_node(
            self._type,
            self.input_arg_names,
            self.output_arg_names,
            name=self._name,
            doc_string=self._doc_string,
            domain=self._domain,
            **attribute_protos,
        )
        return node_proto

    def __str__(self):
        return f"Node: name - {self._name}, type - {self.type}, inputs - {self.input_arg_names}, outputs - {self.output_arg_names}"

    def __repr__(self):
        return str(self)
