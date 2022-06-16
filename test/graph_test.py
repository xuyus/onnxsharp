import onnx
import argparse
from onnxsharp import Model, Graph, Node


def test_subgraph_extraction():
    src = "./testdata/ort_sample_model.onnx"
    model_proto = onnx.load(src)
    m = Model.from_proto(model_proto)

    inputs_of_yield = set()

    def output_filter_func(node: Node):
        if node.type == "YieldOp":
            for i in node.input_arg_names:
                inputs_of_yield.add(i)

    print(inputs_of_yield)
    m._graph.iterate_node(output_filter_func)

    subgraph_inputs = set()

    def input_filter_func(node: Node):
        if node.type == "Gemm" and node.name == "Gemm_0":
            for i in node.input_arg_names:
                subgraph_inputs.add(i)

    print(subgraph_inputs)
    m._graph.iterate_node(input_filter_func)

    subgraph_info = Graph.LogicalSubgraphInfo(
        list(inputs_of_yield),
        list(subgraph_inputs),
    )

    subgraph = Graph.from_logical_subgraph(m._graph, subgraph_info)
    new_m = Model.copy_config(m, subgraph)

    tmp_filename = "/tmp/abc.onnx"
    onnx.save(new_m.to_proto(), tmp_filename)

    model_proto2 = onnx.load(tmp_filename)
    m2 = Model.from_proto(model_proto2)
