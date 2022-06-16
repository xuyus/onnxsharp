
## ONNX Graph Editor

Edit Your ONNX Model With Few Lines of Codes.

Search Nodes:

    model_proto = onnx.load(args.src)
    from onnxsharp import Model, Graph, Node
    m = Model.from_proto(model_proto)
    n = m._graph.get_node("onnx::Cast_565")
    print(n)
    print(m._graph.get_consumer_nodes("onnx::Cast_565"))

Outputs:

> Node: name - Unsqueeze_15, type - Unsqueeze, inputs - ['attention_mask1'], outputs - ['onnx::Cast_565']
> {Node: name - Cast_16, type - Cast, inputs - ['onnx::Cast_565'], outputs - ['onnx::Mul_566']}

Extract SubGraph:

    subgraph_info = Graph.LogicalSubgraphInfo(
        ["202_grad"],
        ["202"],
    )
    subgraph = Graph.from_logical_subgraph(m._graph, subgraph_info)
    new_m = Model.copy_config(m, subgraph)
    onnx.save(new_m.to_proto(), f"pengwa_new_06_15.onnx")


## Installation

pip install -e .
