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

from typing import List, OrderedDict
from onnxsharp import Model, Graph, Node, ValueInfo, NodeArg
from .basics import enforce
import copy
import re


def topological_sort(graph, ops: List[Node]) -> List[Node]:
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
            if graph.is_null(arg_name):
                continue

            if not graph.is_activation(arg_name):
                continue

            j, _ = graph.get_node_with_output_arg_name(arg_name)
            enforce(j is not None, f"Node not found to generate output arg {arg_name}")

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


class LogicalSubgraphInfo(object):
    def __init__(self, g, boundary_output_arg_names, boundary_input_arg_names):
        self._g = g
        self._boundary_output_arg_names = boundary_output_arg_names
        self._boundary_input_arg_names = boundary_input_arg_names

        self._activation_as_subgraph_inputs = []
        self._input_as_subgraph_inputs = []
        self._initializer_as_subgraph_initializers = []
        self._subgraph_nodes = []
        self._output_as_subgraph_outputs = []
        self._activation_as_subgraph_outputs = []


def create_graph_from_logical_subgraph(
    subgraph_info: LogicalSubgraphInfo, drop_initializers=False
):
    g = subgraph_info._g
    extract_sub_graph_nodes(g, subgraph_info, True)
    new_g = Graph()

    print("building nodes....")
    node_count = len(subgraph_info._subgraph_nodes)
    for index, node_name in enumerate(subgraph_info._subgraph_nodes):
        n = g._node_name_mapping[node_name]
        print(f"node>>{index+1} / {node_count}, {n.name} - {n.type}")
        new_n = copy.deepcopy(n)
        new_g.update_node_mapping(new_n)

    new_g._name = g._name

    skip_build_initializer = drop_initializers
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
        # Skip when input_name already in new_g.
        if (
            new_g.is_input(input_name)
            or new_g.is_activation(input_name)
            or new_g.is_initializer(input_name)
        ):
            continue
        n, output_index = g.get_node_with_output_arg_name(input_name)
        value_info: ValueInfo = copy.deepcopy(n.output_arg(output_index)._value_info)
        new_g.add_input(input_name, value_info)

    print("building outputs....")
    for output_arg_name in subgraph_info._output_as_subgraph_outputs:
        value_info: ValueInfo = copy.deepcopy(g._output_map[output_arg_name])
        new_g.add_output(output_arg_name, value_info)

    for output_arg_name in subgraph_info._boundary_output_arg_names:
        if new_g.is_output(output_arg_name):
            continue

        n, output_index = g.get_node_with_output_arg_name(output_arg_name)
        value_info: ValueInfo = copy.deepcopy(n.output_arg(output_index)._value_info)
        new_g.add_output(output_arg_name, value_info)

    for o in set(subgraph_info._activation_as_subgraph_outputs):
        n, output_index = g.get_node_with_output_arg_name(o)
        node_arg: NodeArg = copy.deepcopy(n.output_arg(output_index))
        new_out_name = f"{o}_activation_as_output"
        node_arg._name = new_out_name

        n.replace_output_arg(o, node_arg)

    print("building complete....")

    return new_g


def bfs_from_output(
    g,
    output_arg_names,
    initializer_func,
    input_func,
    activation_func,
    stop_search_level=None,
    stop_search_queue=None,
):
    arg_name_queue = copy.deepcopy(output_arg_names)
    next_level_queue = []
    visited_arg_names = []
    level = 0
    while True:
        # switch to next level.
        if len(arg_name_queue) == 0:
            level += 1

            if stop_search_level is not None and level >= stop_search_level:
                print(
                    "stop search at level",
                    level,
                    (
                        "put next level queue in next_level_queue"
                        if stop_search_queue is not None
                        else ""
                    ),
                )

                if stop_search_queue is not None:
                    stop_search_queue = copy.deepcopy(next_level_queue)
                break

            arg_name_queue = copy.deepcopy(next_level_queue)
            next_level_queue = []

        if len(arg_name_queue) == 0:
            break

        cur_arg_name = arg_name_queue.pop(0)
        if g.is_null(cur_arg_name):
            continue

        if cur_arg_name not in visited_arg_names:
            visited_arg_names.append(cur_arg_name)
        else:
            # skip if arg already be processed.
            # print(
            #     f">>>> [current arg name: {cur_arg_name}] skip since arg {cur_arg_name} already visited"
            # )
            continue

        # handle initializer or input args.
        if g.is_initializer(cur_arg_name) or g.is_input(cur_arg_name):
            if g.is_initializer(cur_arg_name):
                initializer_func(cur_arg_name)

            if g.is_input(cur_arg_name):
                input_func(cur_arg_name)

            continue

        # append input args of current node into queue.
        if g.is_activation(cur_arg_name):
            # handle activation args.
            skip = activation_func(cur_arg_name)
            if skip is True:
                continue

            # append inputs into queue.
            current_node, _ = g.get_node_with_output_arg_name(cur_arg_name)
            for arg_name in current_node.input_arg_names:
                if (
                    arg_name in visited_arg_names
                    or arg_name in arg_name_queue
                    or arg_name in next_level_queue
                ):
                    # skip if arg already processed, or arg already in queue
                    # print(
                    #     f">>>> [current arg name: {cur_arg_name}, owning node: {current_node.name}({current_node.type})] skip adding into queue since arg {arg_name} already visited"
                    # )
                    continue
                # print(
                #     f">>>> [current arg name: {cur_arg_name}, owning node: {current_node.name}({current_node.type})] add input - {arg_name} into queue."
                # )
                next_level_queue.append(arg_name)


def bfs_from_input(g, input_arg_names, output_func, non_output_func):
    arg_name_queue = copy.deepcopy(input_arg_names)
    visited_arg_names = []
    while arg_name_queue:
        cur_arg_name = arg_name_queue.pop(0)
        if g.is_null(cur_arg_name):
            continue

        if cur_arg_name not in visited_arg_names:
            visited_arg_names.append(cur_arg_name)
        else:
            # skip if arg already be processed.
            # print(
            #     f">>>> [current arg name: {cur_arg_name}] skip since arg {cur_arg_name} already visited"
            # )
            continue

        # handle output args.
        if g.is_output(cur_arg_name):
            output_func(cur_arg_name)
            continue

        # handle initializer/input/activation args.
        non_output_func(cur_arg_name)
        # append output args of consumer nodes into queue.
        nodes = g.get_consumer_nodes(cur_arg_name)
        for n in nodes:
            for arg_name in n.output_arg_names:
                if arg_name in visited_arg_names or arg_name in arg_name_queue:
                    # skip if arg already processed, or arg already in queue
                    # print(
                    #     f">>>> [current arg name: {cur_arg_name}, owning node: {n.name}({n.type})] skip adding into queue since arg {arg_name} already visited"
                    # )
                    continue
                # print(
                #     f">>>> [current arg name: {cur_arg_name}, owning node: {n.name}({n.type})] add input - {arg_name} into queue."
                # )
                arg_name_queue.append(arg_name)


def extract_sub_graph_nodes(
    g,
    subgraph_info: LogicalSubgraphInfo,
    strict_input_output_match: bool = False,
):
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
        f">>extract_sub_graph - user given outputs: {output_arg_names}, inputs: {input_arg_names}"
    )

    if strict_input_output_match is True:
        enforce(
            len(output_arg_names) > 0 and len(input_arg_names) > 0,
            "when strict match is enabled, neither of input or output can be empty.",
        )

        reachable_input_arg_names = set()

        def collect_arg_names(arg_name):
            reachable_input_arg_names.add(arg_name)
            return False

        bfs_from_output(
            g,
            output_arg_names,
            collect_arg_names,
            collect_arg_names,
            collect_arg_names,
        )

        # print(f"reachable_input_arg_names: {reachable_input_arg_names}")
        origin_input_arg_names = copy.deepcopy(list(input_arg_names))
        input_arg_names = []
        for i in origin_input_arg_names:
            if i in reachable_input_arg_names and i not in input_arg_names:
                input_arg_names.append(i)

        print(
            f">>extract_sub_graph - strict mode, user given input arg name count: {len(origin_input_arg_names)}, corrected input arg name count: {len(input_arg_names)}"
        )

        enforce(
            len(input_arg_names) > 0,
            "in strict mode, input_arg_names should not be empty.",
        )

        reachable_output_arg_names = set()

        def collect_output_arg_names(arg_name):
            reachable_output_arg_names.add(arg_name)

        bfs_from_input(
            g,
            input_arg_names,
            collect_output_arg_names,
            collect_output_arg_names,
        )

        origin_output_arg_names = copy.deepcopy(list(output_arg_names))
        output_arg_names = []
        for o in origin_output_arg_names:
            if o in reachable_output_arg_names and o not in output_arg_names:
                output_arg_names.append(o)

        print(
            f">>extract_sub_graph - strict mode, user given output arg name count: {len(origin_output_arg_names)}, corrected output arg name count: {len(output_arg_names)}"
        )

        enforce(
            len(output_arg_names) > 0,
            "in strict mode, input_arg_names should not be empty.",
        )

    subgraph_info._boundary_output_arg_names = copy.deepcopy(output_arg_names)
    subgraph_info._boundary_input_arg_names = copy.deepcopy(input_arg_names)

    print(
        f">>extract_sub_graph - refined outputs: {output_arg_names}, inputs: {input_arg_names}"
    )

    enforce(
        len(output_arg_names) == len(set(output_arg_names)),
        "Find duplicated output arg names",
    )
    enforce(
        len(input_arg_names) == len(set(input_arg_names)),
        "Find duplicated input arg names",
    )

    subgraph_nodes: list[str] = []
    initializer_as_subgraph_initializers = []
    activation_as_subgraph_inputs = []
    input_as_subgraph_inputs = []

    def initializer_func(arg_name):
        if strict_input_output_match is True:
            if arg_name in input_arg_names:
                initializer_as_subgraph_initializers.append(arg_name)
        else:
            initializer_as_subgraph_initializers.append(arg_name)
            print(
                f">>>> [current arg name: {arg_name}] skip initializer arg {arg_name}"
            )
        return True

    def input_func(arg_name):
        if strict_input_output_match is True:
            if arg_name in input_arg_names:
                input_as_subgraph_inputs.append(arg_name)
        else:
            input_as_subgraph_inputs.append(arg_name)
            print(
                f">>>> [current arg name: {arg_name}] skip graph input arg {arg_name}"
            )
        return True

    def activation_func(arg_name):
        current_node, _ = g.get_node_with_output_arg_name(arg_name)
        # reach the activation boundary user specified as inputs.
        if arg_name in input_arg_names:
            activation_as_subgraph_inputs.append(arg_name)
            return True  # skip futhur processing.
        elif arg_name in reachable_output_arg_names:
            # Only add the node when it is reachable from outputs.
            subgraph_nodes.append(current_node.name)
            return False
        else:
            return True

    bfs_from_output(g, output_arg_names, initializer_func, input_func, activation_func)

    print(f">>extract_sub_graph - check subgraph node closure. {subgraph_nodes}")
    # For all visited args, besides the output_args, all other args should only be consumed by the nodes in this subgraph.
    output_as_subgraph_outputs = []
    activation_as_subgraph_outputs = []
    for name in subgraph_nodes:
        n = g._node_name_mapping[name]
        for o in n.output_arg_names:
            if g.is_null(o):
                continue
            if g.is_output(o):
                output_as_subgraph_outputs.append(o)

            if o in activation_as_subgraph_outputs:
                continue

            node_output_closure_check = g.all_consumers_of_output_arg_in_subgraph(
                o, subgraph_nodes
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


def safe_remove_subgraph(g, subgraph_info: LogicalSubgraphInfo):
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
        g.remove_output(subgraph_output)

    # Remove nodes in reversed topological order.
    node_count = len(sorted_subgraph_nodes)
    for i in reversed(range(node_count)):
        g.remove_node(sorted_subgraph_nodes[i])

    updated_removable_subgraph_initializers = []
    for subgraph_initializer in removable_subgraph_initializers:
        if subgraph_initializer not in removable_subgraph_inputs:
            updated_removable_subgraph_initializers.append(subgraph_initializer)

    for subgraph_input in removable_subgraph_inputs:
        g.remove_input(subgraph_input)

    for subgraph_initializer in updated_removable_subgraph_initializers:
        g.remove_initializer(subgraph_initializer)

    print("safe_remove_subgraph - successfully remove the subgraph.")


def _create_graph_from_nodes(g: Graph, g_nodes: List[Node]):
    # remove duplications
    g_nodes = list(set(g_nodes))
    updated_node_list = topological_sort(g, g_nodes)

    print("updated_node_list: ", updated_node_list)

    # so far, g_nodes contains 3 level of nodes.

    # Collect all graph-level inputs and initializers.
    new_g = Graph()
    new_g._name = g._name
    new_g._doc_string = g._doc_string

    activation_outs = []
    for n in updated_node_list:
        for index, input_arg_name in enumerate(n.input_arg_names):
            if g.is_input(input_arg_name) and not new_g.is_input(input_arg_name):
                new_g.add_input(
                    input_arg_name, copy.deepcopy(n.input_arg(index)._value_info)
                )

            if g.is_initializer(input_arg_name) and not new_g.is_initializer(
                input_arg_name
            ):
                new_g.add_initializer(
                    input_arg_name,
                    copy.deepcopy(g._initializer_map[input_arg_name]),
                )

        new_g.add_node_copy_from(n, make_non_exist_input_arg_as_graph_input=True)

        for index, output_arg_name in enumerate(n.output_arg_names):
            if g.is_output(output_arg_name) and not new_g.is_output(output_arg_name):
                new_g.add_output(
                    output_arg_name, copy.deepcopy(n.output_arg(index)._value_info)
                )

            if g.is_activation(output_arg_name):
                activation_outs.append([output_arg_name, n.output_arg(index)])

    for output_arg_name, output_arg in activation_outs:
        if new_g.is_output(output_arg_name):
            continue

        new_g.add_output(
            f"{output_arg_name}",
            copy.deepcopy(output_arg._value_info),
        )

    return new_g


def clip_subgraph_around(g: Graph, output_arg_name):
    print(f"clip_subgraph_around>> output_arg_name: {output_arg_name}")
    g_nodes = []

    if g.is_null(output_arg_name):
        raise RuntimeError("output_arg_name is null.")

    if g.is_initializer(output_arg_name):
        raise RuntimeError("output_arg_name is an initializer, stop searching.")

    if g.is_input(output_arg_name):
        raise RuntimeError("output_arg_name is an input, stop searching.")

    n, index = g.get_node_with_output_arg_name(output_arg_name)
    print(f"{n.output_arg(index)}")

    input_level = 3
    cur_level_nodes = [n]
    for i in range(input_level):
        next_level_nodes = []
        for cur_n in set(cur_level_nodes):
            for index, input_arg_name in enumerate(cur_n.input_arg_names):
                if g.is_activation(input_arg_name):
                    n_, index_ = g.get_node_with_output_arg_name(input_arg_name)
                    g_nodes.append(n_)
                    next_level_nodes.append(n_)

        cur_level_nodes = next_level_nodes

    g_nodes.append(n)

    n_consumers = g.get_consumer_nodes(output_arg_name)
    output_level = 3
    cur_level_nodes = [n]
    for i in range(output_level):
        next_level_nodes = []
        for cur_n in set(cur_level_nodes):
            for index, output_arg_name in enumerate(cur_n.output_arg_names):
                if g.is_activation(output_arg_name):
                    n_consumers = g.get_consumer_nodes(output_arg_name)
                    next_level_nodes.extend(n_consumers)
                    g_nodes.extend(n_consumers)

        cur_level_nodes = next_level_nodes

    return _create_graph_from_nodes(g, g_nodes)


def fill_with_execution_plan(g: Graph, file_name):
    # [6] Mul (Mul_17)
    # Free ml-values: (550) onnx::Mul_564
    # Free ml-values: (4971) onnx::Unsqueeze_8287, (4974) onnx::Unsqueeze_8290, (4978) per_input_length_token_715

    # \S: non whiltespace character
    execution_regex = "\[([0-9]+)\] ([\S]+) \(([\S]+)\)"
    free_regex = "Free ml-values: \(([\S]+)\) ([\S]+)"

    with open(file_name) as f:
        for line in f:
            match = re.match(execution_regex, line)
            if match:
                program_counter = int(match.group(1))
                node_type = str(match.group(2))
                node_name = str(match.group(3))
                node = g.get_node_with_name(node_name)

                enforce(node_type == node.type, f"Op type should match for node {node}")
                node._ort_program_counter = program_counter

                continue
            else:
                match = re.match(free_regex, line)
                if match:
                    ortvalue_idx = int(match.group(1))
                    output_arg_name = str(match.group(2))
                    # So far, looks this is not useful
                    continue
                else:
                    print("warning: the line is not parsed correctly ", line)


tag = 0
has_update = False


def auto_cluster_pointwise_graphs(g: Graph):
    global tag
    global has_update
    elementwise_operators = [
        "Abs",
        "Acos",
        "Acosh",
        "Add",
        "And",
        "BiasGelu",
        "Cast",
        "Clip",
        "ConstantOfShape",
        "Div",
        "Equal",
        "Erf",
        "Exp",
        "Greater",
        "Gelu",
        "GeluGrad",
        "Less",
        "Log",
        "MemcpyFromHost",
        "MemcpyToHost",
        "Max",
        "Min",
        "Mul",
        "Neg",
        "Not",
        "Pow",
        "Range",
        "Reshape",
        "Scale",
        "Shape",
        "Sigmoid",
        "SigmoidGrad",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Unsqueeze",
        "Where",
    ]

    node_name_to_tag = {}

    def initialize_node_tag(node: Node):
        global tag
        node_name_to_tag[node.name] = tag
        tag += 1

    def update_flag(old_flag, new_flag):
        has_update = False
        for node_name in node_name_to_tag:
            if node_name_to_tag[node_name] == old_flag:
                node_name_to_tag[node_name] = new_flag
                has_update = True
        return has_update

    g.iterate_node(initialize_node_tag)

    has_update = True
    while has_update:
        has_update = False

        def initialize_node_tag(node: Node):
            global has_update
            if node.type in elementwise_operators:
                # update self's other parents use the same tag.
                # while idx < len(extensible_input_idxs):
                for idx in range(len(node.input_arg_names)):
                    input_arg_name = node.input_arg_names[idx]
                    if g.is_activation(input_arg_name):
                        p_node, _ = g.get_node_with_output_arg_name(
                            node.input_arg_names[idx]
                        )
                        if (
                            p_node.type in elementwise_operators
                            and node_name_to_tag[p_node.name]
                            != node_name_to_tag[node.name]
                        ):
                            has_update = update_flag(
                                node_name_to_tag[p_node.name],
                                node_name_to_tag[node.name],
                            )

        g.iterate_node(initialize_node_tag)

    inversed_map = OrderedDict()
    for name, tag in node_name_to_tag.items():
        if tag not in inversed_map:
            inversed_map[tag] = []

        inversed_map[tag].append(g.get_node_with_name(name))
        print(
            f"append node name {name} into tag {tag}, count become: {len(inversed_map[tag])}"
        )

    g_to_return = OrderedDict()
    for k, v in inversed_map.items():
        if len(v) >= 2:
            print(f"find candidate subgraph with tag: {k}, node count: {len(v)}")
            # subgraphs.append(_create_graph_from_nodes(g, v))

            subgraph_unique_id = unique_id_str(g, v)
            if subgraph_unique_id not in g_to_return:
                g_to_return[subgraph_unique_id] = [
                    _create_graph_from_nodes(g, v),
                    len(v),
                    1,
                ]
            else:
                g_to_return[subgraph_unique_id][2] += 1

    print(f"Find {len(g_to_return)} unique sub graphs.")
    return g_to_return


def unique_id_str(g: Graph, g_nodes: List[Node]):
    g_nodes = list(set(g_nodes))
    sorted_nodes = topological_sort(g, g_nodes)

    unique_id_str = ""
    for n in sorted_nodes:
        input_shapes = []
        for i in range(len(n.input_arg_names)):
            input_shapes.append(str(n.input_arg(i).shape))

        input_shapes_str = ",".join(input_shapes)

        output_shapes = []
        for i in range(len(n.output_arg_names)):
            output_shapes.append(str(n.output_arg(i).shape))

        output_shapes_str = ",".join(output_shapes)

        unique_id_str += f"{len(n.input_arg_names)}.[{input_shapes_str}].{n.type}.{len(n.output_arg_names)}.[{output_shapes_str}]"

    return unique_id_str
