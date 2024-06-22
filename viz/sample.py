from collections import OrderedDict
import copy
from typing import List, Tuple
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
    if int_val >= 0 and int_val <= 9:
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
        index = 36 + ord_v - ord('A')
    elif ord_v >= ord('a') and ord_v <= ord('z'):
        index = 10 + ord_v - ord('a')
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
    offsets = []
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
        offsets.append(start)
        # Increment the start index
        start += len(substring)

    return count, offsets


# print("".join(type_ints))


s = "".join(type_ints)
seq_len = len(s)
print(seq_len)
count_threshold = 2
cluster_threshold = 4
rets = {}


class Pattern:
    def __init__(self, freq, start_offsets, length):
        self.freq = freq
        self.start_offsets = start_offsets
        self.length = length


for i in range(seq_len):
    for j in range(seq_len - 1, i, -1):
        if j - i + 1 < cluster_threshold:
            break

        c, found_offsets = find_substring_appearances(s, s[i:j])

        if c > count_threshold:
            rets[s[i:j]] = Pattern(c, found_offsets, j - i)

            # stop find the substring.
            break


sorted_rets = OrderedDict(
    sorted(rets.items(), key=lambda item: len(item[0]), reverse=True))


print("sorted_rets", [
      f"{'+'.join([get_op_type_from_alphabet(s) for s in k])}: {v.freq}" for k, v in sorted_rets.items()])


def check_range_overlap(range_set_1: List[Tuple[int, int]], lengh_1: int, range_set_2: List[Tuple[int, int]], length_2: int):
    '''For each range in range_set_1, check if it overlaps with any range in range_set_2'''
    for s1 in range_set_1:
        e1 = s1 + lengh_1 - 1
        for s2 in range_set_2:
            e2 = s2 + length_2 - 1
            if not (e1 < s2 or e2 < s1):
                return True

    return False


valid_combinations = {}
for k, v in sorted_rets.items():
    # print("v", v)
    # print('+'.join([get_op_type_from_alphabet(s)
    #       for s in k]), ":", v.length, ':', v.freq)
    offsets = v.start_offsets
    found_be_part_of_others = False
    found_conflict = False
    for k2, v2 in valid_combinations.items():
        # if k in k2 and len(k) < len(k2):
        #     found_be_part_of_others = True

        offset2 = v2.start_offsets

        if check_range_overlap(offsets, v.length, offset2, v2.length) is True:
            found_conflict = True
            # print(f'Conflict: {k} vs {k2}')
            break

    if found_be_part_of_others is False and found_conflict is False:
        valid_combinations[k] = copy.deepcopy(v)


subgraph_starts = {}
for k, v in valid_combinations.items():
    subgraph_str = '+'.join([get_op_type_from_alphabet(s) for s in k])
    # print(subgraph_str, ':', v.freq)
    for x in v.start_offsets:
        subgraph_starts[x] = [v.length, subgraph_str]
# print(rets)


def get_external_inputs_and_outputs_for_subgraph(subgraph: List[onnx.NodeProto]):
    # Loop the node for the given subgraph,
    # 1. if a node output is generated but not used by any other node in current subgraph,
    #     Then this output is subgraph output;
    # 2. if a node input is not generated by any other node in current subgraph,
    #     Then this input is subgraph input;

    all_outputs = set()
    all_inputs = set()
    for node in subgraph:
        for output in node.output:
            all_outputs.add(output)

        for input in node.input:
            all_inputs.add(input)

    subgraph_inputs = []
    subgraph_outputs = []

    for node in subgraph:
        for input in node.input:
            if input not in all_outputs:
                subgraph_inputs.append(input)

        for output in node.output:
            if output not in all_inputs:
                subgraph_outputs.append(output)

    return subgraph_inputs, subgraph_outputs


prefix = """
 <style>

    body {
        font: 300 14px 'Helvetica Neue', Helvetica;
      }

      .node rect {
        stroke: #333;
        fill: #fff;
      }

      .edgePath path {
        stroke: #333;
        fill: #333;
        stroke-width: 1.5px;
      }

  </style>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.7.4/dagre.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.4/dagre-d3.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/5.16.0/d3.js"></script>



  <h2>Graph Visualization</h2>

<svg width='2500' height='2000'>
  <g/>
</svg>

<script>

const start = Date.now();

// Create a new directed graph
var g = new dagre.graphlib.Graph();

// Set an object for the graph label
g.setGraph({});

// Default to assigning a new object as a label for each new edge.
g.setDefaultEdgeLabel(function() { return {}; });

"""


postfix = """
var svg = d3.select("svg"),
    inner = svg.select("g");

// Set up zoom support
var zoom = d3.zoom().on("zoom", function() {
      inner.attr("transform", d3.event.transform);
    });
svg.call(zoom);

middle = Date.now()
const millis = middle - start;
console.log(`moddle elapsed = ${Math.floor(millis / 1000)}`);


// Create the renderer
var render = new dagreD3.render();

// Run the renderer. This is what draws the final graph.
render(inner, g);

after = Date.now()
const millis2 = after - middle;
console.log(`after elapsed = ${Math.floor(millis2 / 1000)}`);

// Center the graph
var initialScale = 0.75;
svg.call(zoom.transform, d3.zoomIdentity.translate(
    (svg.attr("width") - g.graph().width * initialScale) / 2, 20).scale(initialScale));

svg.attr('height', g.graph().height * initialScale + 40);
console.log('end');
end = Date.now()
const millis3 = end - after;
console.log(`end elapsed = ${Math.floor(millis3 / 1000)}`);
</script>


"""


start_node_idx = 0

node_count = len(model_proto.graph.node)
node_output_to_node_map = []
node_protos = []
while start_node_idx < node_count:
    if start_node_idx not in subgraph_starts:
        cur_node = model_proto.graph.node[start_node_idx]
        # v_node_name = cur_node.name
        # for out in cur_node.output:
        #     node_output_to_node_map[out] = v_node_name

        # node_info.append([1, [cur_node], ''])

        node_protos.append(cur_node)

        start_node_idx += 1
        continue

    end_node_idx = start_node_idx + subgraph_starts[start_node_idx][0]

    v_node_name = subgraph_starts[start_node_idx][1]

    print(
        f"count: {subgraph_starts[start_node_idx][0]}, v_node_name: {v_node_name}")

    subgraph = model_proto.graph.node[start_node_idx:end_node_idx + 1]
    subgraph_inputs, subgraph_outputs = get_external_inputs_and_outputs_for_subgraph(
        subgraph)
    # print(subgraph_inputs)
    # print(subgraph_outputs)

    # create a new ONNX node in 'custom' domain, opset 1, optype 'Subgraph',
    # taking subgraph_inputs as inputs and producing subgraph_outputs as outputs.
    # There is no attribute for the subgraph node, so we use an empty attribute list.
    subgraph_node = onnx.helper.make_node(
        'Subgraph', subgraph_inputs, subgraph_outputs, name=v_node_name, domain='custom', opset_version=1, attributes=[])
    node_protos.append(subgraph_node)
    print(f"reduce number of nodes: {subgraph_starts[start_node_idx][0] - 1}")

    start_node_idx = end_node_idx + 1


node_output_to_node_map = {}
print(f"reduced node count to {len(node_protos)}")
for node in node_protos:
    for out in node.output:
        node_output_to_node_map[out] = node

# print(node_output_to_node_map)


# Open a file named gen.html in write mode
f = open("gen2.html", "w")
f.write(prefix + "\n")
# loop through node_protos
for node in node_protos:
    node_str = f"g.setNode('{node.name}', {{ label: '{node.name}', width: 144, height: 100 }});"
    f.write(node_str + "\n")
    for inp in node.input:
        # Ignore the graph input
        if inp not in node_output_to_node_map:
            continue
        # print(
        #     f"edge: {node_output_to_node_map[inp].name} -> {node.name} --> {node.op_type}")
        edge_str = f"g.setEdge('{node_output_to_node_map[inp].name}', '{node.name}');"
        f.write(edge_str + "\n")

    # write the node to the file
    # f.write(str(node) + "\n")
f.write(postfix + "\n")

f.close()
