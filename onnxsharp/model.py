from collections import OrderedDict
from black import validate_cell
import onnx
from onnx import helper, defs, numpy_helper, checker
import copy

from .graph import enforce, Graph, Node, NodeArg, ValueInfo
from .node import Attribute


class Model(object):
    def __init__(self) -> None:
        self._ir_version = None
        self._opset_import = None
        self._producer_name = None
        self._producer_version = None
        self._domain = None
        self._model_version = None
        self._doc_string = None
        self._graph: Graph = None
        self._metadata_props = None

    @classmethod
    def from_proto(self, model_proto):
        m = Model()
        # int64 ir_version = 1;
        m._ir_version = (
            model_proto.ir_version if model_proto.HasField("ir_version") else None
        )

        # repeated OperatorSetIdProto opset_import = 8;
        m._opset_import = model_proto.opset_import

        # string producer_name = 2;
        m._producer_name = (
            model_proto.producer_name if model_proto.HasField("producer_name") else None
        )

        # string producer_version = 3;
        m._producer_version = (
            model_proto.producer_version
            if model_proto.HasField("producer_version")
            else None
        )

        # string domain = 4;
        m._domain = model_proto.domain if model_proto.HasField("domain") else None

        # int64 model_version = 5;
        m._model_version = (
            model_proto.model_version if model_proto.HasField("model_version") else None
        )

        # string doc_string = 6;
        m._doc_string = (
            model_proto.doc_string if model_proto.HasField("doc_string") else None
        )

        # GraphProto graph = 7;
        m._graph = Graph.from_proto(model_proto.graph)

        # repeated StringStringEntryProto metadata_props = 14;
        m._metadata_props = model_proto.metadata_props

        return m

    @classmethod
    def copy_config(cls, m, g):
        new_m = Model()
        new_m._ir_version = m._ir_version
        new_m._opset_import = m._opset_import
        new_m._producer_name = m._producer_name
        new_m._producer_version = m._producer_version
        new_m._domain = m._domain
        new_m._model_version = m._model_version
        new_m._doc_string = m._doc_string
        new_m._graph = g
        new_m._metadata_props = m._metadata_props

        return new_m

    def to_proto(self):
        kwargs = OrderedDict()
        if self._ir_version:
            kwargs["ir_version"] = self._ir_version

        kwargs["opset_imports"] = self._opset_import

        if self._producer_name:
            kwargs["producer_name"] = self._producer_name

        if self._producer_version:
            kwargs["producer_version"] = self._producer_version

        if self._domain:
            kwargs["domain"] = self._domain

        if self._model_version:
            kwargs["model_version"] = self._model_version

        if self._doc_string:
            kwargs["doc_string"] = self._doc_string

        model_proto = helper.make_model(self._graph.to_proto(), **kwargs)
        model_proto.metadata_props.extend(self._metadata_props)
        return model_proto
