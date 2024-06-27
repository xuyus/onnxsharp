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


from typing import Callable, List, OrderedDict, Tuple
from onnxsharp import Graph, Node, NodeArg


from queue import Queue
from collections import defaultdict, deque


def sort_forward_nodes_by_reverse_dfs(
    graph: Graph,
    forward_output_nodes: List[Node],
    shape_size_parents: OrderedDict[int, List[int]],
    nodes_to_execute_before_yieldop: set[Node],
    node_orders: List[Node],
):
    # Note 1: YieldOp is the separator of forward and backward nodes.
    # Note 2: It's also possible some nodes not contributing to the forward output nodes will be
    # executed before YieldOp, for example, if one forward node's output is used by Shape/Size, then
    # the Shape/Size node should be executed before YieldOp to release the memory as soon as possible.

    nodes_in_degree = {}
    to_visit = Queue()

    for n, node_in_degress in nodes_in_degree.items():
        node_in_degress = n.input_edge_count
        nodes_in_degree[n] = node_in_degress
        if node_in_degress == 0:
            to_visit.put(n)

    max_distance = {n: 0 for n in graph.nodes}

    while not to_visit.empty():
        current = to_visit.get()

        if not current:
            continue

        for out_node, _ in current.output_nodes:
            max_distance[out_node] = max(
                max_distance[out_node], max_distance[current] + 1
            )
            nodes_in_degree[out_node] -= 1
            if nodes_in_degree[out_node] == 0:
                to_visit.put(out_node)

    # Reverse DFS from forward output nodes to find all "forward" nodes.
    # The forward nodes are ordered by Reverse DFS traverse.
    graph.reverse_dfs_from(
        forward_output_nodes,
        None,
        lambda n: (
            nodes_to_execute_before_yieldop.add(n),
            node_orders.append(n),
        ),
        lambda n: (
            max_distance[n],
            n.index,
        ),  # The longer distance node should be executed first.
    )

    for parent_node, children_indices in shape_size_parents.items():
        if parent_node not in nodes_to_execute_before_yieldop:
            continue

        for shape_size_node in children_indices:
            if shape_size_node in nodes_to_execute_before_yieldop:
                continue

            try:
                parent_index_position = node_orders.index(parent_node)
            except ValueError:
                raise Exception("Cannot find the parent node in the node orders.")

            node_orders.insert(parent_index_position + 1, shape_size_node)
            nodes_to_execute_before_yieldop.add(shape_size_node)


# Example usage:
# graph = ... # Define your graph here
# is_forward_node = lambda node: ... # Define your criteria for a forward node
# branch_graph_input_nodes, backward_node_in_degree, to_visit = prepare_to_find_branch_graph(graph, is_forward_node)
def prepare_to_find_branch_graph(
    graph: Graph,
    is_forward_node: Callable[[Node], bool],
    branch_graph_input_nodes: List[Node],
    backward_node_in_degree: dict[Node, int],
    to_visit,
):

    for node in graph.nodes:
        # Ignore forward.
        if is_forward_node(node):
            continue

        if node.type == "YieldOp":
            backward_node_in_degree[node] = 0
            to_visit.append(node)
            continue

        input_edge_count = node.input_edge_count
        backward_node_in_degree[node] = input_edge_count

        # A shortcut: input_edge_count could be 0 if it takes graph input directly.
        if input_edge_count == 0:
            branch_graph_input_nodes.append(node)
            continue

        for input_node, _ in node.input_nodes:
            if not input_node:
                continue

            # If the input edge connects to forward nodes, then we remove the in_degree of the node.
            if is_forward_node(input_node):
                input_edge_count -= 1

        backward_node_in_degree[node] = input_edge_count
        if input_edge_count == 0:
            branch_graph_input_nodes.append(node)

    return branch_graph_input_nodes, backward_node_in_degree, to_visit


def find_branch_graph(
    branch_graph_input_nodes: List[Node],
    backward_node_in_degree: List[int],
    branch_graph: List[Node],
    branch_subgraph_consumers: List[Tuple[Node, int]],
    branch_subgraph_outputs: List[NodeArg],
):
    # Loop through the branch_graph_input_nodes to find the branch subgraphs by its output edges in BFS,
    # and find the maximum self_contained subgraph taking the branch_graph_input_nodes as input nodes.
    to_visit_queue = deque()
    in_degree_copy = {k: v for k, v in backward_node_in_degree.items()}
    # backward_node_in_degree.copy()

    # Add all nodes in branch_graph_input_nodes to the queue
    for branch_input_node in branch_graph_input_nodes:

        to_visit_queue.append(branch_input_node)
        branch_graph.append(branch_input_node)

    while to_visit_queue:
        current = to_visit_queue.popleft()

        if not current:
            continue

        for node_it, _ in current.output_nodes:

            in_degree_copy[node_it] -= 1

            if in_degree_copy[node_it] == 0:
                to_visit_queue.append(node_it)
                branch_graph.append(node_it)

    # At this point, branch_graph is a big subgraph that contains all the nodes that are purely
    # triggered by the branch_graph_input_nodes, other graph input/initializers and leaf nodes (for example Constant).
    for n in branch_graph:
        if n.output_edge_count == 0:
            # In case the node connect to graph outputs or nothings, append all outputs as the branch subgraph outputs.
            for output_def in n._output_args:
                branch_subgraph_outputs.append(output_def)
            continue

        for output_node, port_to_port in n.output_nodes:
            dest_in_port = port_to_port[1]
            if output_node not in branch_graph:
                branch_subgraph_consumers.append((output_node, dest_in_port))
                branch_subgraph_outputs.append(n._output_args[port_to_port[0]])


class GroupNode:
    def __init__(self, node_list: List[Node]):
        self.nodes = node_list
        self.is_outputted = False
        self.input_args = []
        self.output_args = []
        intermediate_args = {}

        for node in self.nodes:
            for arg in node._input_args:
                if arg not in intermediate_args:
                    if arg not in self.input_args:
                        self.input_args.append(arg)

            for arg in node._output_args:
                intermediate_args[arg] = True

            if node.output_edge_count == 0:
                for arg in node._output_args:
                    if arg not in self.output_args:
                        self.output_args.append(arg)
                continue

            for output_node, port_to_port in node.output_nodes:
                # Only if the output arg is used by nodes outside the group, then it is an output arg.
                if output_node not in self.nodes:
                    out_arg = node._output_args[port_to_port[0]]
                    if out_arg not in self.output_args:
                        self.output_args.append(out_arg)

        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print(
        #     f"GroupNode: {self.nodes[0].name} ==> {[a.name for a in self.input_args]} ==> {[a.name for a in self.output_args]}"
        # )
        # print("--------------------------------------------------")
        # print(",".join([n.name for n in self.nodes]))
        # print(
        #     "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        # )


def tag_node_to_associated_outputs(
    graph: Graph,
    nodes_to_execute_before_yieldop: set[Node],
    branch_subgraph_outputs: List[NodeArg],
    branch_graph: List[Node],
    group_node_collection: List[GroupNode],
    output_arg_to_grouped_node: dict[str, GroupNode],
):
    node_to_its_associated_outputs = {}
    handled_branch_subgraph_end_nodes = set()
    for output_arg in branch_subgraph_outputs:
        end_node, _ = graph.get_node_with_output_arg_name(output_arg.name)

        if not end_node or end_node in handled_branch_subgraph_end_nodes:
            continue

        handled_branch_subgraph_end_nodes.add(end_node)

        end_nodes = [end_node]
        graph.reverse_dfs_from(
            end_nodes,
            None,
            lambda n: node_to_its_associated_outputs.setdefault(n, set()).add(
                output_arg.name
            ),
            None,
            lambda _, to: to in nodes_to_execute_before_yieldop,
        )

    associated_outputs_to_nodes = {}
    for node in branch_graph:
        associated_outputs = node_to_its_associated_outputs.get(node, set())
        associated_outputs_to_nodes.setdefault(
            frozenset(sorted(associated_outputs)), []
        ).append(node)

    for associated_outputs, nodes in associated_outputs_to_nodes.items():
        group_node_collection.append(GroupNode(nodes))
        grouped_node = group_node_collection[-1]
        for output_arg in grouped_node.output_args:
            output_arg_to_grouped_node[output_arg.name] = grouped_node


def update_backward_in_degree(
    backward_node_in_degree: dict[Node, int],
    branch_subgraph_consumers: List[Tuple[Node, int]],
):
    """
    For each GroupNode, its execution is non-blocking main critical path rooting from YieldOp.
    The only dependencies of a GroupNode are either graph input/initializer/forward nodes, or
    the output nodes of another GroupNode.
    So we treat those GroupNode(s) as a single unit that can be executed anytime when it is
    firstly needed by the main critical path.
    """
    for output_node, dest_in_port in branch_subgraph_consumers:
        assert (
            backward_node_in_degree[output_node] > 0
        ), "Backward node in-degree must be greater than 0"
        backward_node_in_degree[output_node] -= 1


def output_grouped_nodes(
    graph: Graph,
    output_arg_name: str,
    output_arg_to_grouped_node: dict[str, GroupNode],
    node_orders: List[Node],
    topo_order: List[Node],
):
    # Ensure the output_arg is in the output_arg_to_grouped_node dictionary
    assert (
        output_arg_name in output_arg_to_grouped_node
    ), f"output_arg_to_grouped_node does not contain output_arg named {output_arg_name}"

    # Get the grouped node from the dictionary
    grouped_node = output_arg_to_grouped_node[output_arg_name]

    # If the grouped node is already outputted, return
    if grouped_node.is_outputted:
        return

    # Iterate over input arguments of the grouped node
    for input_arg in grouped_node.input_args:
        # If the input argument does not exist, continue to the next one
        if not input_arg.name:
            continue

        # If the input argument is in the dictionary and not yet outputted, call the function recursively
        if (
            input_arg.name in output_arg_to_grouped_node
            and not output_arg_to_grouped_node[input_arg.name].is_outputted
        ):
            output_grouped_nodes(
                graph,
                input_arg.name,
                output_arg_to_grouped_node,
                node_orders,
                topo_order,
            )

    # Add the nodes to the node orders and topological order
    for n in grouped_node.nodes:
        node_orders.append(n)
        topo_order.append(n)

    # Mark the grouped node as outputted
    grouped_node.is_outputted = True


def memory_efficient_topological_sort(
    graph: Graph,
    node_orders: List[Node],
):
    yield_op: Node = None
    shape_size_parents: dict[Node, List[Node]] = {}
    for node in graph.nodes:
        if node.type == "YieldOp":
            yield_op = node

        if node.type in ["Shape", "Size"] and node.input_edge_count > 0:
            input_nodes_and_ports = node.input_nodes
            assert (
                len(input_nodes_and_ports) == 1
            ), "Shape/Size node should have only one input."
            input_node = input_nodes_and_ports[0][0]
            if input_node not in shape_size_parents:
                shape_size_parents[input_node] = []

            shape_size_parents[input_node].append(node)

    # Firstly, sort the forward nodes with customized ReverseDFS.
    num_nodes = len(graph.nodes)
    forward_output_nodes = []

    if yield_op:
        for input_node, _ in yield_op.input_nodes:
            if input_node:
                forward_output_nodes.append(input_node)
    else:
        for output in graph.output_names:
            output_node, _ = graph.get_node_with_output_arg_name(output)
            forward_output_nodes.append(output_node)

    # Create a set for cheaper search.
    nodes_to_execute_before_yieldop: set[Node] = set()

    sort_forward_nodes_by_reverse_dfs(
        graph,
        forward_output_nodes,
        shape_size_parents,
        nodes_to_execute_before_yieldop,
        node_orders,
    )

    if num_nodes == len(node_orders):
        assert (
            len(nodes_to_execute_before_yieldop) == num_nodes
        ), "All nodes should be executed before YieldOp."
        return

    # Secondly, sort the backward nodes with customized Kahn's algorithm.
    num_of_backward_nodes = num_nodes - len(node_orders)
    backward_node_in_degree: dict[Node, int] = {}
    topo_order = []
    to_visit = deque()

    def is_forward_op(node):
        return node in nodes_to_execute_before_yieldop

    branch_graph_input_nodes = []

    prepare_to_find_branch_graph(
        graph,
        is_forward_op,
        branch_graph_input_nodes,
        backward_node_in_degree,
        to_visit,
    )

    branch_graph = []
    branch_subgraph_consumers = []
    branch_subgraph_outputs = []
    find_branch_graph(
        branch_graph_input_nodes,
        backward_node_in_degree,
        branch_graph,
        branch_subgraph_consumers,
        branch_subgraph_outputs,
    )

    # Cluster the nodes in the branch_graph based on the associated outputs.
    group_node_collection = []
    output_arg_to_grouped_node: dict[str, GroupNode] = {}
    tag_node_to_associated_outputs(
        graph,
        nodes_to_execute_before_yieldop,
        branch_subgraph_outputs,
        branch_graph,
        group_node_collection,
        output_arg_to_grouped_node,
    )

    update_backward_in_degree(backward_node_in_degree, branch_subgraph_consumers)

    while len(to_visit) > 0:
        # current = to_visit.get()
        current = to_visit.popleft()

        if not current:
            continue

        for input_node, port_to_port in current.input_nodes:
            if not input_node:
                continue

            input_arg = current._input_args[port_to_port[1]]
            if not input_arg.name:
                continue

            grouped_node = output_arg_to_grouped_node.get(input_arg.name)
            if grouped_node and not grouped_node.is_outputted:
                output_grouped_nodes(
                    graph,
                    input_arg.name,
                    output_arg_to_grouped_node,
                    node_orders,
                    topo_order,
                )

        node_orders.append(current)

        for out_node, _ in current.output_nodes:
            # out_node = output_edge.node
            backward_node_in_degree[out_node] -= 1
            if backward_node_in_degree[out_node] == 0:
                to_visit.append(out_node)

        topo_order.append(current)

    # For the group nodes that are not outputted, we need to output them.
    left_output_arg_to_grouped_node_vector = []
    for output_arg_name, grouped_node in output_arg_to_grouped_node.items():
        if not grouped_node.is_outputted:
            left_output_arg_to_grouped_node_vector.append(
                (output_arg_name, grouped_node)
            )

    if left_output_arg_to_grouped_node_vector:
        # Sort to ensure deterministic order.
        left_output_arg_to_grouped_node_vector.sort(key=lambda pair: pair[0])

        for output_arg_name, grouped_node in left_output_arg_to_grouped_node_vector:
            if not grouped_node.is_outputted:
                output_grouped_nodes(
                    graph,
                    output_arg_name,
                    output_arg_to_grouped_node,
                    node_orders,
                    topo_order,
                )

    if num_of_backward_nodes != len(topo_order):
        # for n in graph.nodes:
        #     if is_forward_op(n):
        #         continue
        #     if n not in topo_order:
        #         print(n.name)
        raise Exception(
            f"Some nodes for backward are not included in the topological sort: "
            f"{num_of_backward_nodes} vs {len(topo_order)}"
        )
