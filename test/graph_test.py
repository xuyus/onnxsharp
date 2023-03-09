import onnx
import pytest

from onnxsharp import (
    Model,
    Node,
    LogicalSubgraphInfo,
    create_graph_from_logical_subgraph,
    auto_cluster_pointwise_graphs,
)

src = "./testdata/ort_sample_model.onnx"


def test_subgraph_extraction():
    m = Model.load_model(src)

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

    subgraph_info = LogicalSubgraphInfo(
        m._graph,
        list(inputs_of_yield),
        list(subgraph_inputs),
    )

    subgraph = create_graph_from_logical_subgraph(subgraph_info)
    new_m = Model.copy_config(m, subgraph)
    new_m.save_model("extract_subgraph.onnx")


def test_clip_subgraph_from_output_arg():
    from onnxsharp import Model, clip_subgraph_around

    m = Model.load_model(src)
    new_g = clip_subgraph_around(m._graph, "onnx::Gemm_6")
    new_m = Model.copy_config(m, new_g)
    new_m.save_model("clipped_subgraph.onnx")


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("with_execution_plan", [False, True])
def test_node_print(level, with_execution_plan):
    from onnxsharp import Model, fill_with_execution_plan

    m = Model.load_model(src)
    # Optionally load execution plan exported by ORT.
    # fill_with_execution_plan(m._graph, "testdata/execution_plan.log")

    m._graph.summarize_nodes(level, with_execution_plan=with_execution_plan)


@pytest.mark.parametrize(
    "model_path",
    [
        "./testdata/ort_sample_model.onnx",
    ],
)
def test_tensor_print(model_path):
    from onnxsharp import Model

    m = Model.load_model(model_path)
    m._graph.summarize_tensors()

    level = 0
    m._graph.summarize_nodes(level)


def test_node_include_shape_print(level=0):
    from onnxsharp import Model, fill_with_execution_plan

    m = Model.load_model(src)
    # Optionally load execution plan exported by ORT.
    # fill_with_execution_plan(m._graph, "testdata/execution_plan.log")

    m._graph.summarize_nodes(level, include_shape=True)


def test_desc_graph_inputs():
    from onnxsharp import Model

    m = Model.load_model(src)
    m._graph.summarize_inputs()


def test_auto_clustering():
    m = Model.load_model(src)
    rets = auto_cluster_pointwise_graphs(m._graph)
    prefix = "ort_sample_"
    idx = 0
    for _, subgraph in rets.items():
        dest = f"{prefix}.{subgraph[1]}nodes-{subgraph[2]}freq-{idx}.onnx"
        new_m = Model.copy_config(m, subgraph[0])
        onnx.save(new_m.to_proto(), dest)
        idx += 1


def test_save_model():
    from onnxsharp import Model

    m = Model.load_model(src)
    m.save_model("./saved_model.onnx")
    m.save_model("./saved_model_external_data.onnx", save_as_external_data=True)
    m.save_model_to_string("./saved_model.txt")
