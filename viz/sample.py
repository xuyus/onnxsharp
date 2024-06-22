import onnx


# load the model using onnx.load
# model_proto = onnx.load(
#     "C:\\Users\\pengwa\\dev\\onnx-sharp\\test\\clipped_subgraph.onnx"
# )
model_proto = onnx.load(
    R"C:\Users\pengwa\models\mistral_bingads\0522\0523_exp_ort_flash_attention_2_2048_run_001_execution_model_training.onnx"
)


node_output_to_node_map = {}
for node in model_proto.graph.node:
    for output in node.output:
        node_output_to_node_map[output] = node


types = []

op_type_to_int = [
    "ATen",
    "Add",
    "BroadcastGradientArgs",
    "Cast",
    "Concat",
    "ConcatTraining",
    "CumSum",
    "Div",
    "Equal",
    "Expand",
    "FlattenAndUnpad",
    "FusedMatMul",
    "Gather",
    "GatherElements",
    "Gemm",
    "MemcpyFromHost",
    "MemcpyToHost",
    "Mul",
    "Neg",
    "NonZero",
    "Pad",
    "PadAndUnflatten",
    "PythonOp",
    "PythonOpGrad",
    "QuickGelu",
    "QuickGeluGrad",
    "Range",
    "ReduceSum",
    "Reshape",
    "Shape",
    "SimplifiedLayerNormalization",
    "SimplifiedLayerNormalizationGrad",
    "Slice",
    "SliceGrad",
    "SoftmaxCrossEntropyLossInternal",
    "SoftmaxCrossEntropyLossInternalGrad",
    "Split",
    "SplitTraining",
    "Squeeze",
    "Sub",
    "Sum",
    "Transpose",
    "Unsqueeze",
    "Where",
    "YieldOp",
]


def get_alphabet_from_int(int_val: int):
    return chr(int_val + ord("a"))


type_ints = []
# loop through the model_proto.graph.node
for node in model_proto.graph.node:
    node_str = f"g.setNode('{node.name}', {{ label: '{node.name}', width: 144, height: 100 }});"
    types.append(node.op_type)
    ascii_int = op_type_to_int.index(node.op_type)
    type_ints.append(get_alphabet_from_int(ascii_int))
    print(f"{node.op_type} -> {ascii_int} -> {get_alphabet_from_int(ascii_int)}")
    for inp in node.input:
        # Ignore the graph input
        if inp not in node_output_to_node_map:
            continue
        edge_str = f"g.setEdge('{node_output_to_node_map[inp].name}', '{node.name}');"


def find_substring_appearances(long_string, substring):
    # Initialize count
    count = 0
    # Initialize start index
    start = 0

    # Loop until the substring is found in the long_string
    while start < len(long_string):
        # Find the next index of the substring
        start = long_string.find(substring, start)
        if start == -1:  # If the substring is not found, break the loop
            break
        # Increment the count
        count += 1
        # Increment the start index
        start += len(substring)

    return count


print("".join(type_ints))


s = "".join(type_ints)
seq_len = len(s)
print(seq_len)
count_threshold = 2
cluster_threshold = 16
rets = {}


for i in range(seq_len):
    for j in range(seq_len - 1, i, -1):
        if j - i + 1 < cluster_threshold:
            break

        c = find_substring_appearances(s, s[i:j])

        if c > count_threshold:
            rets[s[i:j]] = c

            # stop find the substring.
            break


print(rets)
