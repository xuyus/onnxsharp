from collections import OrderedDict
from typing import List, Tuple
from black import validate_cell
import onnx
from onnx import helper, defs, numpy_helper, checker
import copy
import numpy as np

from .node import enforce, Node, NodeArg, ValueInfo
from .tensor import TensorType, Tensor, TensorShape


class Graph(object):
    def __init__(self) -> None:
        self._output_arg_name_to_node_mapping: OrderedDict[str, Node] = OrderedDict()
        self._node_name_mapping: OrderedDict[str, Node] = OrderedDict()

        self._name = None
        self._initializer_map: OrderedDict[str, Tensor] = OrderedDict()
        self._doc_string = None

        self._input_map: OrderedDict[str, ValueInfo] = OrderedDict()
        self._output_map: OrderedDict[str, ValueInfo] = OrderedDict()

    @classmethod
    def from_proto(cls, graph_proto):
        g = Graph()
        # repeated NodeProto node = 1;
        for node_proto in graph_proto.node:
            n = Node.from_proto(g, node_proto)
            g.update_node_mapping(n)

        # string name = 2;
        g._name = graph_proto.name

        # repeated TensorProto initializer = 5;
        for initializer in graph_proto.initializer:
            g._initializer_map[initializer.name] = Tensor.from_proto(initializer)

        # string doc_string = 10;
        g._doc_string = graph_proto.doc_string

        # The inputs and outputs of the graph.
        # repeated ValueInfoProto input = 11;
        for value_info_proto in graph_proto.input:
            g._input_map[value_info_proto.name] = ValueInfo.from_proto(value_info_proto)

        # repeated ValueInfoProto output = 12;
        for value_info_proto in graph_proto.output:
            g._output_map[value_info_proto.name] = ValueInfo.from_proto(
                value_info_proto
            )

        # Information for the values in the graph. The ValueInfoProto.name's
        # must be distinct. It is optional for a value to appear in value_info list.
        # repeated ValueInfoProto value_info = 13;
        value_info_map = OrderedDict()
        for value_info_proto in graph_proto.value_info:
            value_info_name = value_info_proto.name
            value_info_map[value_info_name] = ValueInfo.from_proto(value_info_proto)

        # Update input/output args according to available ValueInfos.
        for _, node in g._node_name_mapping.items():
            for node_input in node.input_arg_names:
                if node_input in value_info_map:
                    node.replace_input_arg(
                        node_input, NodeArg.from_valueinfo(value_info_map[node_input])
                    )
                elif node_input in g._input_map:
                    node.replace_input_arg(
                        node_input, NodeArg.from_valueinfo(g._input_map[node_input])
                    )

            for node_output in node.output_arg_names:
                if node_output in value_info_map:
                    node.replace_output_arg(
                        node_output, NodeArg.from_valueinfo(value_info_map[node_output])
                    )
                elif node_output in g._output_map:
                    node.replace_output_arg(
                        node_output,
                        NodeArg.from_valueinfo(g._output_map[node_output]),
                    )
            # since here we did not change the output name, so following line is not required here.
            # g.update_node_mapping(node)

        return g

    def update_node_mapping(self, new_node: Node):
        self._node_name_mapping[new_node.name] = new_node
        for index, o in enumerate(new_node.output_arg_names):
            self._output_arg_name_to_node_mapping[o] = (new_node, index)

    def get_tensor(self, output_arg_name: str) -> Tensor:
        enforce(output_arg_name is not None, "output_arg_name should not be None")
        enforce(
            output_arg_name in self._initializer_map,
            f"output_arg_name {output_arg_name} not exists",
        )
        return self._initializer_map[output_arg_name]

    def get_node_with_output_arg_name(self, output_arg_name: str) -> Tuple[Node, int]:
        enforce(output_arg_name is not None, "output_arg_name should not be None")
        enforce(
            output_arg_name in self._output_arg_name_to_node_mapping,
            f"output_arg_name {output_arg_name} not exists",
        )
        return self._output_arg_name_to_node_mapping[output_arg_name]

    def get_node_with_name(self, node_name: str) -> Node:
        enforce(node_name is not None, "node_name should not be None")
        enforce(
            node_name in self._node_name_mapping,
            f"node name {node_name} not exists",
        )
        return self._node_name_mapping[node_name]

    def get_consumer_nodes(self, output_arg_name: str):
        nodes: set[Node] = set()
        for _, n in self._node_name_mapping.items():
            if output_arg_name in n.input_arg_names:
                nodes.add(n)

        return nodes

    def iterate_node(self, func):
        for _, n in self._node_name_mapping.items():
            func(n)

    @property
    def input_names(self):
        return self._input_map.keys()

    @property
    def output_names(self):
        return self._output_map.keys()

    def is_input(self, output_arg_name: str) -> bool:
        return output_arg_name in self.input_names

    def is_initializer(self, output_arg_name: str) -> bool:
        return output_arg_name in self._initializer_map.keys()

    def is_activation(self, output_arg_name: str) -> bool:
        return output_arg_name in self._output_arg_name_to_node_mapping.keys()

    def is_output(self, output_arg_name: str) -> bool:
        return output_arg_name in self.output_names

    def is_null(self, output_arg_name: str) -> bool:
        return output_arg_name == ""

    def replace_input_arg(self, existing_input_arg_name, new_input_arg: NodeArg):
        enforce(
            not self.is_null(existing_input_arg_name),
            "existing_input_arg_name cannot be null",
        )
        enforce(
            self.is_activation(existing_input_arg_name),
            "existing_input_arg_name must be input of node",
        )
        consumers = self.get_consumer_nodes(existing_input_arg_name)
        for c in consumers:
            c.replace_input_arg(existing_input_arg_name, new_input_arg)

    def remove_output(self, output_arg_name):
        enforce(
            self.is_output(output_arg_name),
            f"remove_output cannot remove non graph output.",
        )
        enforce(
            len(self.get_consumer_nodes(output_arg_name)) == 0,
            "graph output is expected to NOT consumed by any nodes.",
        )
        del self._output_map[output_arg_name]

    def remove_node(self, node_name):
        enforce(
            node_name in self._node_name_mapping, f"no node found named {node_name}"
        )
        output_arg_names = copy.deepcopy(
            [str(n) for n in self._node_name_mapping[node_name].output_arg_names]
        )
        for o in output_arg_names:
            enforce(
                self.is_output(o) is False,
                f"pls remove arg {o} from graph output before removing the node.",
            )
            enforce(
                self.is_activation(o) and len(self.get_consumer_nodes(o)) == 0,
                f"node output {o} is expected to NOT consumed by any nodes.",
            )

        for o in output_arg_names:
            del self._output_arg_name_to_node_mapping[o]

        del self._node_name_mapping[node_name]

    def add_input(self, input_arg_name, value_info: ValueInfo):
        enforce(value_info is not None, "value_info should not be None")
        enforce(not self.is_null(input_arg_name), f"{input_arg_name} is null")
        enforce(
            not self.is_input(input_arg_name),
            f"{input_arg_name} already exists as input.",
        )
        enforce(
            not self.is_activation(input_arg_name),
            f"{input_arg_name} already exists as activation.",
        )
        enforce(
            not self.is_output(input_arg_name),
            f"{input_arg_name} already exists as output.",
        )

        self._input_map[input_arg_name] = value_info

    def add_initializer(self, input_arg_name, tensor: Tensor):
        enforce(tensor is not None, "tensor should not be None")
        enforce(not self.is_null(input_arg_name), f"{input_arg_name} is null")
        enforce(
            not self.is_initializer(input_arg_name),
            f"{input_arg_name} already exists as graph initializer.",
        )
        enforce(
            not self.is_activation(input_arg_name),
            f"{input_arg_name} already exists as activation.",
        )
        enforce(
            not self.is_output(input_arg_name),
            f"{input_arg_name} already exists as output.",
        )

        # TODO(pengwa): add value_info check between input and initializer.
        # if self.is_input(input_arg_name):
        #     enforce(self._input_map[input_arg_name] == tensor.)

        self._initializer_map[input_arg_name] = tensor

    # output_arg_name MUST be an activation before we make it a graph outputs.
    # Correspondingly, removing from graph outputs, it is still an activation.
    def add_output(self, output_arg_name, value_info: ValueInfo):
        enforce(value_info is not None, "value_info should not be None")
        enforce(not self.is_null(output_arg_name), f"{output_arg_name} is null")
        enforce(
            not self.is_input(output_arg_name),
            f"{output_arg_name} already exists as input.",
        )
        enforce(
            not self.is_output(output_arg_name),
            f"{output_arg_name} already exists as output.",
        )
        enforce(
            not self.is_initializer(output_arg_name),
            f"{output_arg_name} already exists as initializer.",
        )
        enforce(
            self.is_activation(output_arg_name),
            f"{output_arg_name} must exist as activation before adding as output.",
        )

        self._output_map[output_arg_name] = value_info

    def add_node_copy_from(
        self, node: Node, make_non_exist_input_arg_as_graph_input=False
    ):
        print(f"add_node_copy_from>>>entering for node {node}.")
        for output_arg_name in node.output_arg_names:
            enforce(
                not self.is_activation(output_arg_name)
                and not self.is_initializer(output_arg_name)
                and not self.is_input(output_arg_name)
                and not self.is_output(output_arg_name),
                f"add_node_copy_from>>> node output arg {output_arg_name} already exists in graph.",
            )

        input_candidates = OrderedDict()
        for input_arg_index, input_arg_name in enumerate(node.input_arg_names):
            if self.is_null(input_arg_name):
                continue

            arg_exist = (
                self.is_activation(input_arg_name)
                or self.is_initializer(input_arg_name)
                or self.is_input(input_arg_name)
            )

            if arg_exist is False:
                if make_non_exist_input_arg_as_graph_input is False:
                    enforce(
                        arg_exist,
                        f"add_node_copy_from>>> node arg {input_arg_name} not exists in graph.",
                    )
                else:
                    input_candidates[input_arg_name] = copy.deepcopy(
                        node.input_arg(input_arg_index)._value_info
                    )

        for name, value_info in input_candidates.items():
            print(
                f"add_node_copy_from>>> making non-exist arg {name} as graph input: {value_info}"
            )
            self.add_input(name, value_info)

        n = copy.deepcopy(node)
        self.update_node_mapping(n)
        print(f"add_node_copy_from>>>exiting for node {n}.")

    def add_node(
        self,
        type: str,
        name: str,
        input_arg_names: List[str],
        output_arg_names: List[str],
        domain: str,
        doc_string: str,
        **kwargs,
    ):
        for input_arg_name in input_arg_names:
            enforce(
                not self.is_output(input_arg_name),
                f"{input_arg_name} already exists as output.",
            )
            enforce(
                self.is_activation(input_arg_name)
                or self.is_input(input_arg_name)
                or self.is_initializer(input_arg_name),
                f"{input_arg_name} already exists as activation/input/initializer.",
            )

        for output_arg_name in output_arg_names:
            enforce(
                not self.is_input(output_arg_name),
                f"{output_arg_name} already exists as input.",
            )
            enforce(
                not self.is_output(output_arg_name),
                f"{output_arg_name} already exists as output.",
            )
            enforce(
                not self.is_initializer(output_arg_name),
                f"{output_arg_name} already exists as initializer.",
            )
            enforce(
                self.is_activation(output_arg_name),
                f"{output_arg_name} already exists as activation.",
            )

        n = Node()
        n._g = self
        n._name = name
        n._type = type
        n._input_args = [NodeArg(i) for i in input_arg_names]
        n._output_args = [NodeArg(o) for o in output_arg_names]
        n._domain = domain

        for attr_name, attr_value in kwargs.items():
            n._attr[attr_name] = attr_value

        n._doc_string = doc_string
        self.update_node_mapping(n)
        return n

    def remove_input(self, input_arg_name):
        enforce(
            self.is_input(input_arg_name), "remove_input cannot remove non graph input."
        )
        enforce(
            input_arg_name in self._input_map,
            f"no graph input arg found named {input_arg_name}",
        )
        enforce(
            len(self.get_consumer_nodes(input_arg_name)) == 0,
            "cannot remove graph input {input_arg_name} consumed by other nodes.",
        )
        del self._input_map[input_arg_name]
        if self.is_initializer(input_arg_name):
            del self._initializer_map[input_arg_name]

    def remove_initializer(self, input_arg_name):
        enforce(
            not self.is_input(input_arg_name),
            "the initializer is also grah input, pls use remove_input to remove graph input.",
        )
        enforce(
            self.is_initializer(input_arg_name),
            "remove_initializer cannot remove non graph initializer.",
        )
        enforce(
            len(self.get_consumer_nodes(input_arg_name)) == 0,
            "cannot remove graph input {input_arg_name} consumed by other nodes.",
        )
        del self._initializer_map[input_arg_name]

    def to_proto(self):
        graph_proto = onnx.GraphProto()

        node_protos = []
        value_info_protos = []
        for name, node in self._node_name_mapping.items():
            node_protos.append(node.to_proto())
            for output_arg in node._output_args:
                if (
                    output_arg._value_info is not None
                    and output_arg._value_info.has_type()
                ):
                    value_info_protos.append(output_arg.to_proto())
        graph_proto.node.extend(node_protos)

        graph_proto.name = self._name

        initializer_protos = []
        for name, tensor in self._initializer_map.items():
            initializer_protos.append(tensor.to_proto())
        graph_proto.initializer.extend(initializer_protos)

        graph_proto.doc_string = self._doc_string

        input_protos = []
        for name, value_info in self._input_map.items():
            input_protos.append(value_info.to_proto())
        graph_proto.input.extend(input_protos)

        output_protos = []
        for name, value_info in self._output_map.items():
            enforce(value_info is not None, f"value_info is null for {name}")
            output_protos.append(value_info.to_proto())
        graph_proto.output.extend(output_protos)

        graph_proto.value_info.extend(value_info_protos)

        return graph_proto

    def all_consumers_of_output_arg_in_subgraph(self, arg_name, subgraph_nodes):
        consumers = self.get_consumer_nodes(arg_name)
        for c in consumers:
            if c not in subgraph_nodes:
                print(
                    f"found external node [{c.name}({c.type})], consuming output_arg {arg_name}"
                )
                return False

        return True

    def summarize_inputs(self):
        import pprint

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self._input_map)

    def summarize_tensors(self):
        import pprint

        # https://en.wikipedia.org/wiki/Half-precision_floating-point_format

        smallest_subnormal_number = 5.96e-8
        smallest_normal_number = 6.10e-5
        largest_norm_number = 65504

        pp = pprint.PrettyPrinter(indent=4)
        nan_tensor_names: OrderedDict[str, Tensor] = OrderedDict()
        inf_tensor_names: OrderedDict[str, Tensor] = OrderedDict()

        def _summarize_tensors(tensors_map):
            for initializer_name, tensor in tensors_map:
                if np.isnan(tensor.value).any():
                    if initializer_name not in nan_tensor_names:
                        nan_tensor_names[initializer_name] = tensor
                    continue

                if np.isinf(tensor.value).any():
                    if initializer_name not in nan_tensor_names:
                        inf_tensor_names[initializer_name] = tensor
                    continue

                subnormal_candidiates = np.logical_and(
                    np.abs(tensor.value) > 0,
                    np.abs(tensor.value) <= smallest_subnormal_number,
                )
                if subnormal_candidiates.any():
                    to_print = tensor.value
                    if tensor.value.ndim > 0:
                        indices = np.where(subnormal_candidiates)
                        to_print = tensor.value[indices]

                    print(
                        f"Warning: find a tensor {initializer_name} having subnormal number {to_print} around fp16 lower boundary."
                    )

                around_fp16_boundary = np.abs(tensor.value) >= largest_norm_number
                if around_fp16_boundary.any():
                    to_print = tensor.value
                    if tensor.value.ndim > 0:
                        indices = np.where(around_fp16_boundary)
                        to_print = tensor.value[indices]

                    print(
                        f"Warning: find a tensor {initializer_name} having number {to_print} around fp16 upper boundary."
                    )

        _summarize_tensors(self._initializer_map.items())

        constant_tensors: OrderedDict[str, Tensor] = OrderedDict()
        for _, n in self._node_name_mapping.items():
            if n.type == "Constant":
                constant_tensors[n.output_arg_names[0]] = Tensor.from_proto(
                    n._attr["value"].value
                )

        _summarize_tensors(constant_tensors.items())

        if len(nan_tensor_names) == 0 and len(inf_tensor_names) == 0:
            print("NaN or inf not found for all initializers.")
        else:
            print(
                f"Found {len(nan_tensor_names)} initializers contains Nan. Be cautious used for training."
            )

            print(
                f"Found {len(inf_tensor_names)} initializers contains Inf. Be cautious used for training."
            )
        # pp.pprint(self._initializer_map.items())

    def summarize_nodes(
        self, level=0, with_excution_plan=False, include_shape=False, op_type=None
    ):
        import pprint

        pp = pprint.PrettyPrinter(indent=4)

        def _get_node_pattern(n: Node, cur_level):
            optypestr_list = []
            if cur_level < level:
                for i in n.input_arg_names:
                    if not self.is_activation(i):
                        continue

                    p_node, _ = self.get_node_with_output_arg_name(i)
                    node_str = _get_node_pattern(p_node, cur_level + 1)
                    optypestr_list.append(node_str)

            types_str = ",".join(optypestr_list)
            execution_str = (
                "@" + str(n._ort_program_counter)
                if with_excution_plan is True and n._ort_program_counter is not None
                else ""
            )
            bw_str = (
                "_B"
                if n._doc_string is not None and "Backward pass" in n._doc_string
                else ""
            )

            shape_str = ""
            if include_shape is True:
                all_input_shape_str = ",".join(
                    [
                        "(" + ",".join([str(s) for s in node_input_arg.shape]) + ")"
                        if node_input_arg.shape
                        else "None"
                        for node_input_arg in n._input_args
                    ]
                )
                shape_str = f"<-[{all_input_shape_str}]"
            return f"{n.type}{bw_str}{execution_str}({types_str}){shape_str}"

        op_type_str_summary: OrderedDict[str, int] = OrderedDict()
        for name, node in self._node_name_mapping.items():
            if op_type is not None and op_type != node.type:
                continue

            pattern_str = _get_node_pattern(node, 0)

            if pattern_str not in op_type_str_summary:
                op_type_str_summary[pattern_str] = 0

            op_type_str_summary[pattern_str] += 1

        sorted_tuples = sorted(
            op_type_str_summary.items(), key=lambda item: item[1], reverse=True
        )

        print(f"## {level} levels of node summary:")
        pp.pprint(sorted_tuples)

        return
