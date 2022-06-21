
## ONNX Graph Editor

Edit Your ONNX Model With Few Lines of Codes.

Search Nodes:

    model_proto = onnx.load(args.src)
    from onnxsharp import Model, Graph, Node, LogicalSubgraphInfo,
        create_graph_from_logical_subgraph,
    m = Model.from_proto(model_proto)
    n, out_index = m._graph.get_node_with_output_arg_name("onnx::Cast_565")
    print(n)
    print(m._graph.get_consumer_nodes("onnx::Cast_565"))

Outputs:

> Node: name - Unsqueeze_15, type - Unsqueeze, inputs - ['attention_mask1'], outputs - ['onnx::Cast_565']
> {Node: name - Cast_16, type - Cast, inputs - ['onnx::Cast_565'], outputs - ['onnx::Mul_566']}

Add Node/Inputs/Initializers/Outputs:

    m._graph.add_node()
    m._graph.add_input()
    m._graph.add_initializer()
    m._graph.add_output()


Extract SubGraph:

    subgraph_info = LogicalSubgraphInfo(
        m._graph,
        ["202_grad"],
        ["202"],
    )
    subgraph =create_graph_from_logical_subgraph(subgraph_info)
    new_m = Model.copy_config(m, subgraph)
    onnx.save(new_m.to_proto(), f"pengwa_new_06_15.onnx")

Clip SubGraph Around Output Edges:

    new_g = clip_subgraph_around(m._graph, "onnx::Gemm_6")
    new_m = Model.copy_config(m, new_g)
    dest = f"clipped_subgraph.onnx"
    onnx.save(new_m.to_proto(), dest)

## Installation

pip install -e .
