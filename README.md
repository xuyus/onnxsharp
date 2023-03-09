
## ONNX Graph Tool Box

Operate on your ONNX model with ease.

Load/Save model:

```python
from onnxsharp import Model

m = Model.load_model("model.onnx") # An alternative: Model.from_proto(onnx.load("model.onnx"))
...

new_m.save_model(f"new_model.onnx") # An alternative: onnx.save(new_m.to_proto(), f"new_model.onnx")

```

Access Graph APIs:

```python
# Search Nodes:
n, out_index = m._graph.get_node_with_output_arg_name("onnx::Cast_565")
print(n)
print(m._graph.get_consumer_nodes("onnx::Cast_565"))
# Outputs:
# > Node: name - Unsqueeze_15, type - Unsqueeze, inputs - ['attention_mask1'], outputs - ['onnx::Cast_565']
# > {Node: name - Cast_16, type - Cast, inputs - ['onnx::Cast_565'], outputs - ['onnx::Mul_566']}

# Summarize Tensors with FP16 range check:
# Both initializers (of type TensorProto) and Constant node's value attribute (of type TensorProto) will be checked.
# Can be used as a data range safety check, especially some of those tensors will be represented as FP16
# in training.
m._graph.summarize_tensors()


# Clip SubGraph Around Output Edges:
new_g = clip_subgraph_around(m._graph, "onnx::Gemm_6")
new_m = Model.copy_config(m, new_g)
new_m.save_model(f"clipped_subgraph.onnx")


# Summarize Nodes with different level:
# Optionally load execution plan exported by ORT.
# This will show execution program counter for each operator when summarize patterns.
# fill_with_execution_plan(m._graph, "testdata/execution_plan.log")

# level = 0: single node as pattern.
# level = 1: node and its first level parents as a pattern.
# level = 2: node and its first level parents, and second level parents (e.g. parents of the first level parents) as a pattern.
m._graph.summarize_nodes(level, with_execution_plan=with_execution_plan)

# Outputs:
# > 0 levels of node summary:
#     [('Gemm()', 5),
#     ('ReduceSum()', 2),
#     ('Relu()', 1),
#     ('YieldOp()', 1),
#     ('ReluGrad()', 1)]

# > 2 levels of node summary:
#     [('Gemm()', 1),
#     ('Relu(Gemm())', 1),
#     ('Gemm(Relu(Gemm()))', 1),
#     ('YieldOp(Gemm(Relu()))', 1),
#     ('Gemm(YieldOp(Gemm()))', 1),
#     ('ReluGrad(Gemm(YieldOp()),Relu(Gemm()))', 1),
#     ('Gemm(ReluGrad(Gemm(),Relu()))', 1),
#     ('ReduceSum(ReluGrad(Gemm(),Relu()))', 1),
#     ('Gemm(YieldOp(Gemm()),Relu(Gemm()))', 1),
#     ('ReduceSum(YieldOp(Gemm()))', 1)]

# > graph_test.py::test_node_include_shape_print ## 0 levels of node summary:
#     [('Gemm()<-[(input1_dim0,input1_dim1),(500,784),(500)]', 1),
#     ('Relu()<-[(input1_dim0,500)]', 1),
#     ('Gemm()<-[(input1_dim0,500),(10,500),(10)]', 1),
#     ('YieldOp()<-[(input1_dim0,10)]', 1),
#     ('Gemm_B()<-[None,(10,500)]', 1),
#     ('ReluGrad_B()<-[(input1_dim0,500),(input1_dim0,500)]', 1),
#     ('Gemm_B()<-[(input1_dim0,500),(input1_dim0,input1_dim1)]', 1),
#     ('ReduceSum_B()<-[(input1_dim0,500),None]', 1),
#     ('Gemm_B()<-[None,(input1_dim0,500)]', 1),
#     ('ReduceSum_B()<-[None,None]', 1)]



# Extract SubGraph:
subgraph_info = LogicalSubgraphInfo(
    m._graph,
    ["202_grad"],
    ["202"],
)
subgraph =create_graph_from_logical_subgraph(subgraph_info)
new_m = Model.copy_config(m, subgraph)
new_m.save_model(f"new_model.onnx")


# Add Node/Inputs/Initializers/Outputs:
m._graph.add_node()
m._graph.add_input()
m._graph.add_initializer()
m._graph.add_output()

```

## Installation

Dev Install

    pip install -e .

Install from PyPI

    pip install --upgrade onnxsharp
