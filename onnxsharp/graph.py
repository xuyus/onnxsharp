from collections import OrderedDict
from black import validate_cell
import onnx
from onnx import helper, defs, numpy_helper, checker
import copy

from .node import enforce, Node, NodeArg, ValueInfo
from .tensor import TensorType, Tensor, TensorShape


class Graph(object):
    def __init__(self) -> None:
        self._output_arg_name_to_node_mapping: OrderedDict[str, Node] = OrderedDict()
        self._node_name_mapping: OrderedDict[str, Node] = OrderedDict()

        self._name = None
        self._initializer_map = OrderedDict()
        self._doc_string = None

        self._input_map = OrderedDict()
        self._output_map = OrderedDict()

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
            g._initializer_map[initializer.name] = Tensor(initializer)

        # string doc_string = 10;
        g._doc_string = graph_proto.doc_string

        # // The inputs and outputs of the graph.
        # repeated ValueInfoProto input = 11;
        for value_info_proto in graph_proto.input:
            g._input_map[value_info_proto.name] = ValueInfo.from_proto(value_info_proto)

        # repeated ValueInfoProto output = 12;
        for value_info_proto in graph_proto.output:
            g._output_map[value_info_proto.name] = ValueInfo.from_proto(
                value_info_proto
            )

        # // Information for the values in the graph. The ValueInfoProto.name's
        # // must be distinct. It is optional for a value to appear in value_info list.
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

    def get_node(self, output_arg_name: str) -> Node:
        n, _ = self.get_node_with_index(output_arg_name)
        return n

    def get_node_with_index(self, output_arg_name: str) -> tuple[Node, int]:
        enforce(output_arg_name is not None, "output_arg_name should not be None")
        enforce(
            output_arg_name in self._output_arg_name_to_node_mapping,
            f"output_arg_name {output_arg_name} not exists",
        )
        return self._output_arg_name_to_node_mapping[output_arg_name]

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
                # todo: clean up this if
                if output_arg._value_info is not None:
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

    class LogicalSubgraphInfo(object):
        def __init__(self, boundary_output_arg_names, boundary_input_arg_names) -> None:
            self._boundary_output_arg_names = boundary_output_arg_names
            self._boundary_input_arg_names = boundary_input_arg_names

            self._activation_as_subgraph_inputs = []
            self._input_as_subgraph_inputs = []
            self._initializer_as_subgraph_initializers = []
            self._subgraph_nodes = []
            self._output_as_subgraph_outputs = []
            self._activation_as_subgraph_outputs = []

    @classmethod
    def from_logical_subgraph(cls, g, subgraph_info: LogicalSubgraphInfo):
        g.extract_sub_graph_nodes(subgraph_info)
        new_g = Graph()

        print("building nodes....")
        node_count = len(subgraph_info._subgraph_nodes)
        for index, node_name in enumerate(subgraph_info._subgraph_nodes):
            n = g._node_name_mapping[node_name]
            print(f"node>>{index+1} / {node_count}, {n.name} - {n.type}")
            new_n = copy.deepcopy(n)
            new_g.update_node_mapping(new_n)

        new_g._name = g._name

        skip_build_initializer = True
        if skip_build_initializer is False:
            print("building initializers....")
            for initializer_name in subgraph_info._initializer_as_subgraph_initializers:
                new_initializer = copy.deepcopy(g._initializer_map[initializer_name])
                new_g._initializer_map[initializer_name] = new_initializer

        new_g._doc_string = g._doc_string

        print("building inputs....")
        for input_name in subgraph_info._input_as_subgraph_inputs:
            value_info: ValueInfo = copy.deepcopy(g._input_map[input_name])
            print(f"{input_name} - {value_info}")
            new_g.add_input(input_name, value_info)

        for input_name in subgraph_info._activation_as_subgraph_inputs:
            (n, output_index) = g.get_node_with_index(input_name)
            value_info: ValueInfo = copy.deepcopy(
                n.output_args(output_index)._value_info
            )
            new_g.add_input(input_name, value_info)

        print("building outputs....")
        for output_arg_name in subgraph_info._output_as_subgraph_outputs:
            value_info: ValueInfo = copy.deepcopy(g._output_map[output_arg_name])
            new_g.add_output(output_arg_name, value_info)

        for o in set(subgraph_info._activation_as_subgraph_outputs):
            (n, output_index) = g.get_node_with_index(o)
            value_info: ValueInfo = copy.deepcopy(
                n.output_args(output_index)._value_info
            )

            print(f"add activation {o} as graph output {value_info}")
            new_g.add_output(o, value_info)

        return new_g

    def extract_sub_graph_nodes(self, subgraph_info: LogicalSubgraphInfo):
        """Return nodes of subgraph ending with dest_node.
        Args:
            dest_node: output node of the subgraph to find
            input_checker: customized input check function: bool func(node)
        Return:
            a set of nodes
        """
        output_arg_names = subgraph_info._boundary_output_arg_names
        input_arg_names = subgraph_info._boundary_input_arg_names

        print(
            f">>extract_sub_graph - outputs: {output_arg_names}, inputs: {input_arg_names}"
        )

        enforce(
            len(output_arg_names) == len(set(output_arg_names)),
            "Find duplicated output arg names",
        )
        enforce(
            len(input_arg_names) == len(set(input_arg_names)),
            "Find duplicated input arg names",
        )

        arg_name_queue = copy.deepcopy(output_arg_names)
        visited_arg_names = []

        subgraph_nodes: list[str] = []
        initializer_as_subgraph_initializers = []
        activation_as_subgraph_inputs = []
        input_as_subgraph_inputs = []
        while arg_name_queue:
            cur_arg_name = arg_name_queue.pop(0)
            if self.is_null(cur_arg_name):
                continue

            if cur_arg_name not in visited_arg_names:
                visited_arg_names.append(cur_arg_name)
            else:
                # skip if arg already be processed.
                print(
                    f">>>> [current arg name: {cur_arg_name}] skip since arg {cur_arg_name} already visited"
                )
                continue

            if self.is_initializer(cur_arg_name) or self.is_input(cur_arg_name):
                if self.is_initializer(cur_arg_name):
                    initializer_as_subgraph_initializers.append(cur_arg_name)
                    print(
                        f">>>> [current arg name: {cur_arg_name}] skip initializer arg {cur_arg_name}"
                    )

                if self.is_input(cur_arg_name):
                    input_as_subgraph_inputs.append(cur_arg_name)
                    print(
                        f">>>> [current arg name: {cur_arg_name}] skip graph input arg {cur_arg_name}"
                    )

                continue

            # append input args of current node into queue.
            if self.is_activation(cur_arg_name):
                # reach the activation boundary user specified as inputs.
                if cur_arg_name in input_arg_names:
                    activation_as_subgraph_inputs.append(cur_arg_name)
                    continue

                current_node = self.get_node(cur_arg_name)
                subgraph_nodes.append(current_node.name)

                for arg_name in current_node.input_arg_names:
                    if arg_name in visited_arg_names or arg_name in arg_name_queue:
                        # skip if arg already processed, or arg already in queue
                        print(
                            f">>>> [current arg name: {cur_arg_name}, owning node: {current_node.name}({current_node.type})] skip adding into queue since arg {arg_name} already visited"
                        )
                        continue
                    print(
                        f">>>> [current arg name: {cur_arg_name}, owning node: {current_node.name}({current_node.type})] add input - {arg_name} into queue."
                    )
                    arg_name_queue.append(arg_name)

        print(f">>extract_sub_graph - check subgraph node closure.")
        # For all visited args, besides the output_args, all other args should only be consumed by the nodes in this subgraph.
        output_as_subgraph_outputs = []
        activation_as_subgraph_outputs = []
        for name in subgraph_nodes:
            n = self._node_name_mapping[name]
            for o in n.output_arg_names:
                if self.is_null(o):
                    continue
                if self.is_output(o):
                    output_as_subgraph_outputs.append(o)

                if o in activation_as_subgraph_outputs:
                    continue

                node_output_closure_check = (
                    self.all_consumers_of_output_arg_in_subgraph(o, subgraph_nodes)
                )

                if node_output_closure_check is False:
                    # add this output arg as output boundary.
                    activation_as_subgraph_outputs.append(o)

        subgraph_info._activation_as_subgraph_inputs = activation_as_subgraph_inputs
        subgraph_info._input_as_subgraph_inputs = input_as_subgraph_inputs
        subgraph_info._activation_as_subgraph_outputs = activation_as_subgraph_outputs
        subgraph_info._output_as_subgraph_outputs = output_as_subgraph_outputs
        subgraph_info._initializer_as_subgraph_initializers = (
            initializer_as_subgraph_initializers
        )
        subgraph_info._subgraph_nodes = subgraph_nodes
        print(f">>extract_sub_graph - completed withour error")

    def safe_remove_subgraph(self, subgraph_info: LogicalSubgraphInfo):
        removable_subgraph_inputs = subgraph_info._owning_graph_inputs_safe_to_remove
        removable_subgraph_initializers = (
            subgraph_info._owning_graph_initializers_safe_to_remove
        )
        subgraph_outputs = subgraph_info._owning_graph_outputs_safe_to_remove
        sorted_subgraph_nodes = subgraph_info._owning_graph_nodes_safe_to_remove

        print(
            f">>safe_remove_subgraph - outputs: {subgraph_outputs}, inputs: {removable_subgraph_inputs}"
        )

        # Remove output arg from graph outputs first.
        for subgraph_output in subgraph_outputs:
            self.remove_output(subgraph_output)

        # Remove nodes in reversed topological order.
        node_count = len(sorted_subgraph_nodes)
        for i in reversed(range(node_count)):
            self.remove_node(sorted_subgraph_nodes[i])

        updated_removable_subgraph_initializers = []
        for subgraph_initializer in removable_subgraph_initializers:
            if subgraph_initializer not in removable_subgraph_inputs:
                updated_removable_subgraph_initializers.append(subgraph_initializer)

        for subgraph_input in removable_subgraph_inputs:
            self.remove_input(subgraph_input)

        for subgraph_initializer in updated_removable_subgraph_initializers:
            self.remove_initializer(subgraph_initializer)

        print("safe_remove_subgraph - successfully remove the subgraph.")

    def topological_sort(self, ops: list[Node]) -> list[Node]:
        """Topological sort of graph."""
        # sort by name, the result will be reversed alphabeta
        ops.sort(key=lambda op: op.name)

        def _push_stack(stack, node, in_stack):
            stack.append(node)
            if node in in_stack:
                raise ValueError("Graph has cycles, node.name=%r." % ops[node].name)
            in_stack[node] = True

        def _get_unvisited_child(g, node, not_visited):
            for child in g[node]:
                if child in not_visited:
                    return child
            return -1

        n = len(ops)

        # a list of consumer node indices.
        g = [[] for _ in range(n)]
        op_name_to_index = {}
        for i, op in enumerate(ops):
            op_name_to_index[op.name] = i

        for i, op in enumerate(ops):
            input_arg_names = sorted(set(op.input_arg_names))
            for arg_name in input_arg_names:
                if self.is_null(arg_name):
                    continue

                if not self.is_activation(arg_name):
                    continue

                j = self.get_node(arg_name)
                enforce(
                    j is not None, f"Node not found to generate output arg {arg_name}"
                )

                if j.name not in op_name_to_index:
                    # this is a temp fix
                    continue
                enforce(j.name in op_name_to_index, f"Node {j.name} not exist.")
                g[op_name_to_index[j.name]].append(i)

        # label for each op. highest = sink nodes.
        label = [-1 for _ in range(n)]
        stack = []
        in_stack = dict()
        not_visited = dict.fromkeys(range(n))
        label_counter = n - 1

        while not_visited:
            node = list(not_visited.keys())[0]
            _push_stack(stack, node, in_stack)
            while stack:
                node = _get_unvisited_child(g, stack[-1], not_visited)
                if node != -1:
                    _push_stack(stack, node, in_stack)
                else:
                    node = stack.pop()
                    in_stack.pop(node)
                    not_visited.pop(node)
                    label[node] = label_counter
                    label_counter -= 1

        ret = [x for _, x in sorted(zip(label, ops))]
        return ret
