import onnx


# load the model using onnx.load
model_proto = onnx.load(
    R"C:\dev\onnxsharp\0523_exp_ort_flash_attention_2_2048_run_001_execution_model_training.onnx"
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
    if int_val >=0 and int_val <= 9:
        return chr(int_val + ord("0"))
    elif int_val >= 10 and int_val <= 35:
        return chr(int_val - 10 + ord("a"))
    elif int_val >= 36 and int_val <= 61:
        return chr(int_val - 36 + ord("A"))
    raise ValueError(f'unsupported int_val: {int_val}')

def get_op_type_from_alphabet(val: str):
    ord_v = ord(val)
    index = None
    if ord_v >= ord('0') and ord_v <= ord('9'): 
        index = ord_v - ord('0')
    elif ord_v >= ord('A') and ord_v <= ord('Z'):
        index =  36 + ord_v - ord('A')
    elif ord_v >= ord('a') and ord_v <= ord('z'):
        index =  10 + ord_v - ord('a')
    else:
        raise ValueError(f'fail to convert {val}. ord_v: {ord_v}')

    return op_type_to_int[index]

type_ints = []
# loop through the model_proto.graph.node
for node in model_proto.graph.node:
    # node_str = f"g.setNode('{node.name}', {{ label: '{node.name}', width: 144, height: 100 }});"
    # types.append(node.op_type)
    ascii_int = op_type_to_int.index(node.op_type)
    type_ints.append(get_alphabet_from_int(ascii_int))
    # print(f"{node.op_type} -> {ascii_int} -> {get_alphabet_from_int(ascii_int)}")
    # for inp in node.input:
    #     # Ignore the graph input
    #     if inp not in node_output_to_node_map:
    #         continue
    #     edge_str = f"g.setEdge('{node_output_to_node_map[inp].name}', '{node.name}');"


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


# print("".join(type_ints))


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
            rets[s[i:j]] = [c, [i, j]]

            # stop find the substring.
            break

import copy

from collections import OrderedDict

sorted_rets = OrderedDict(sorted(rets.items(), key=lambda item: len(item[0]), reverse=True))


valid_combinations = {}
for k, v in sorted_rets.items():
    start, end = v[1]
    found_be_part_of_others = False
    found_conflict = False
    for k2, v2 in valid_combinations.items():
        if k in k2 and len(k) < len(k2):
           found_be_part_of_others = True

        s2, e2 = v2[1]
        if not (e2 < start or end < s2):
            found_conflict = True

        

        

    if found_be_part_of_others is False and found_conflict is False:
       valid_combinations[k] = copy.deepcopy(v)

           
subgraph_starts = {}
for k, v in valid_combinations.items():
    freq, (x, y) = v
    subgraph_str = '+'.join([get_op_type_from_alphabet(s) for s in k])
    print(subgraph_str, ':', freq)
    subgraph_starts[x] = y - x
# print(rets)

# for idx, node in enumerate(model_proto.graph.node):

#     if idx in subgraph_starts:





