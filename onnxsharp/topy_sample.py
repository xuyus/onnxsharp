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
