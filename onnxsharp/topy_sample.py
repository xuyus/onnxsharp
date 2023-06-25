def topological_sort(GraphProto graph, ops: List[NodeProto], bool is_subgraph=False) -> List[NodeProto]:
    """Topological sort of graph."""
    # sort by name, the result will be reversed alphabeta
    ops.sort(key=lambda op: op.name)

    graph_inputs: List[str] = graph.input
    graph_initializer: List[str] = graph.initializer
    graph_outputs = graph.output
    
    arg_name_to_node_map = {}
    for op in ops:
        for o_str in op.output:
            arg_name_to_node_map[o_str] = op
            
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
            if arg_name in arg_name_to_node_map:
                j = arg_name_to_node_map[arg_name]
                g[op_name_to_index[j.name]].append(i)
            elif arg_name in graph_inputs or arg_name in graph_initializer:
                continue
            elif is_subgraph:
                # todo check from parent graph
                continue
            else:
                raise RuntimeError()

                
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
