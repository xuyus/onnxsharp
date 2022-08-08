import onnx


def how():
    all_funcs = {
        "select_model": "Select the model to work on.",
        "sum_nodes": "Summarize nodes using level (Default: 0), include_shapes (Default: False) and op_type (which must exist, Default: None).",
        "list_nodes": "List nodes with specified operator types.",
        "clip_graph": "Clip the graph using node output argument name.",
    }
    print(f"Available functions: {all_funcs}")


global _M


def _check_model():
    global _M
    from .basics import enforce

    enforce(_M is not None, "Please select model before any other commands.")


def select_model(src):
    from onnxsharp import (
        Model,
        Node,
        LogicalSubgraphInfo,
        create_graph_from_logical_subgraph,
    )

    global _M
    model_proto = onnx.load(src)
    _M = Model.from_proto(model_proto)
    print(f"Model loaded successfully from {src}.")


def sum_nodes(level=0, include_shape=False, op_type=None):
    _check_model()
    global _M
    _M._graph.summarize_nodes(0, include_shape=include_shape, op_type=op_type)


def list_nodes(op_type):
    _check_model()
    from onnxsharp import (
        Model,
        Node,
        LogicalSubgraphInfo,
        create_graph_from_logical_subgraph,
    )

    global _M

    def output_filter_func(node: Node):
        if node.type == op_type:
            print(node)

    _M._graph.iterate_node(output_filter_func)


def clip_graph(out_arg_name):
    _check_model()
    from onnxsharp import Model, Graph, Node, clip_subgraph_around

    global _M
    n = out_arg_name
    new_g = clip_subgraph_around(_M._graph, n)
    new_m = Model.copy_config(_M, new_g)
    new_n = n.replace(":", "_", -1)
    dest = f"{new_n}.onnx"
    onnx.save(new_m.to_proto(), dest)

    print(f"model saved to {dest}")
