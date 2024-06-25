from collections import OrderedDict
import copy
from typing import List, Tuple
import onnx


# load the model using onnx.load
model_proto = onnx.load(
    # R"C:\dev\onnxsharp\0523_exp_ort_flash_attention_2_2048_run_001_execution_model_training.onnx"
    # R"C:\Users\pengwa\models\mistral_bingads\0522\0523_exp_ort_flash_attention_2_2048_run_001_execution_model_training.onnx"
    # R"C:\Users\pengwa\dev\onnx-sharp\bingads_model_0_optimized_pre_grad_training.onnx"
    R"C:\Users\pengwa\dev\onnx-sharp\bingads_model_0_execution_model_training.onnx"
    # R"C:\Users\pengwa\dev\onnx-sharp\test\clipped_subgraph.onnx"
)


def get_alphabet_from_int(int_val: int):
    if int_val >= 0 and int_val <= 9:
        return chr(int_val + ord("0"))
    elif int_val >= 10 and int_val <= 35:
        return chr(int_val - 10 + ord("a"))
    elif int_val >= 36 and int_val <= 61:
        return chr(int_val - 36 + ord("A"))
    raise ValueError(f"unsupported int_val: {int_val}")


op_type_to_int = []
for node in model_proto.graph.node:
    if node.op_type not in op_type_to_int:
        op_type_to_int.append(node.op_type)


def get_op_type_from_alphabet(val: str):
    ord_v = ord(val)
    index = None
    if ord_v >= ord("0") and ord_v <= ord("9"):
        index = ord_v - ord("0")
    elif ord_v >= ord("A") and ord_v <= ord("Z"):
        index = 36 + ord_v - ord("A")
    elif ord_v >= ord("a") and ord_v <= ord("z"):
        index = 10 + ord_v - ord("a")
    else:
        raise ValueError(f"fail to convert {val}. ord_v: {ord_v}")

    return op_type_to_int[index]


type_ints = []
# loop through the model_proto.graph.node
for node in model_proto.graph.node:
    ascii_int = op_type_to_int.index(node.op_type)
    type_ints.append(get_alphabet_from_int(ascii_int))


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


s = "".join(type_ints)
seq_len = len(s)
print(seq_len)
count_threshold = 2
cluster_threshold = 8
rets = {}


class Pattern:
    def __init__(self, freq, start_offsets, length):
        self.freq = freq
        self.start_offsets = copy.deepcopy(start_offsets)
        self.length = length


has_update = True
placeholder_index = 0

placeholderindex_to_pattern = []

transform_sequences = []


def _get_subgraph_str_from_encoded_str(subgraph_encoded_str: str):
    return "+".join([get_op_type_from_alphabet(s) for s in subgraph_encoded_str])


print(s)
print(_get_subgraph_str_from_encoded_str(s))

# First round, find the most longest repeated subgraph pattern.
repeated_pattern = {}
for i in range(seq_len):
    for j in range(seq_len, i, -1):
        length = j - i + 1
        if length < cluster_threshold:
            continue

        cur_substr = s[i : j + 1]

        if (
            length % 2 == 0
            and cur_substr[i + length // 2 + 1 : -1] == s[i : i + length // 2]
        ):
            print("skip the pattern contained repeated pattern")
            continue

        c, found_offsets = find_substring_appearances(s, cur_substr)

        if c >= count_threshold:
            repeated_pattern[cur_substr] = Pattern(c, found_offsets, length)
            # print(f"found repated pattern: {cur_substr}, {c}, {length}")
            break

sorted_repeated_pattern = OrderedDict(
    sorted(
        repeated_pattern.items(),
        key=lambda item: item[1].length * 10000 + item[1].freq,
        reverse=True,
    )
)


def check_range_overlap(
    range_set_1: List[Tuple[int, int]],
    lengh_1: int,
    range_set_2: List[Tuple[int, int]],
    length_2: int,
):
    """For each range in range_set_1, check if it overlaps with any range in range_set_2"""
    for s1 in range_set_1:
        e1 = s1 + lengh_1 - 1
        for s2 in range_set_2:
            e2 = s2 + length_2 - 1
            if not (e1 < s2 or e2 < s1):
                return True

    return False


valid_combinations = {}
for k, v in sorted_repeated_pattern.items():
    # print("v", v)
    # print('+'.join([get_op_type_from_alphabet(s)
    #       for s in k]), ":", v.length, ':', v.freq)
    offsets = v.start_offsets
    found_conflict = False
    for k2, v2 in valid_combinations.items():
        offset2 = v2.start_offsets

        if check_range_overlap(offsets, v.length, offset2, v2.length) is True:
            found_conflict = True
            # print(f'Conflict: {k} vs {k2}')
            break

    if found_conflict is False:
        valid_combinations[k] = copy.deepcopy(v)


for k, v in valid_combinations.items():
    if v.freq >= count_threshold:
        cur_substr = k
        s = s.replace(cur_substr, "|" * len(cur_substr))
        rets[cur_substr] = Pattern(v.freq, v.start_offsets, v.length)
        print(f"apply valid_combinations: {cur_substr}, {v.freq}, {v.length}")

while has_update is True:
    has_update = False
    seq_len = len(s)

    # print(f"new iteration >>>>>> seq_len: {seq_len}, s: {s}")
    i = 0
    while i < seq_len:
        if s[i] == "|":
            i += 1
            continue

        j = seq_len - 1
        while j > i:
            if s[j] == "|":
                j -= 1
                continue

            length = j - i + 1
            if length < cluster_threshold:
                break

            cur_substr = s[i : j + 1]
            start = cur_substr.find("|")
            if start != -1:
                j = i + start - 1
                continue

            assert "|" not in cur_substr, f"cur_substr: {cur_substr}, {i}, {j}"

            c, found_offsets = find_substring_appearances(s, cur_substr)

            if c >= count_threshold:
                s = s.replace(cur_substr, "|" * len(cur_substr))
                rets[cur_substr] = Pattern(c, found_offsets, length)
                has_update = True
                # print(
                #     f"found: {_get_subgraph_str_from_encoded_str(cur_substr)}, {c}, {length}"
                # )
                # stop find the substring.
                break
            j -= 1

        if has_update:
            break

        i += 1

print(f"s: {s}")

sorted_rets = OrderedDict(sorted(rets.items(), key=lambda item: item[0], reverse=True))


subgraph_starts = {}

for k, v in sorted_rets.items():
    subgraph_str = k  # _get_subgraph_str_from_encoded_str(k)
    # print(subgraph_str, ':', v.freq)
    for x in v.start_offsets:
        subgraph_starts[x] = [v.length, subgraph_str]

sorted_subgraph_starts = OrderedDict(
    sorted(subgraph_starts.items(), key=lambda item: item[0], reverse=False)
)


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

    <!-- Pull in JQuery dependencies -->
    <link rel="stylesheet" href="tipsy.css">
    <script src="https://code.jquery.com/jquery-1.9.1.min.js"></script>
    <script src="tipsy.js"></script>

    <h2>Graph Visualization</h2>

    <svg width='2500' height='2000'>
    <g/>
    </svg>

    <script>


    // Create a new directed graph
    var g = new dagre.graphlib.Graph();

    // Set an object for the graph label
    g.setGraph({});

    // Default to assigning a new object as a label for each new edge.
    g.setDefaultEdgeLabel(function() { return {}; });

    


"""


postfix = """

    // Add states to the graph, set labels, and style
    Object.keys(states).forEach(function(state) {
        var value = states[state];
        value.label = state;
        value.rx = value.ry = 5;
        g.setNode(state, value);
    });

    var svg = d3.select("svg"),
        inner = svg.append("g"); //svg.select("g");

    // Set up zoom support
    var zoom = d3.zoom().on("zoom", function() {
        inner.attr("transform", d3.event.transform);
        });
    svg.call(zoom);



    // Create the renderer
    var render = new dagreD3.render();

    // Run the renderer. This is what draws the final graph.
    render(inner, g);

    // Simple function to style the tooltip for the given node.
    var styleTooltip = function(name, description) {
        return "<p class='name'>" + name + "</p><p class='description'>" + description + "</p>";
    };

    inner.selectAll("g.node")
        .attr("title", function(v) { return styleTooltip(v, g.node(v).description) })
        .each(function(v) { $(this).tipsy({ gravity: "w", opacity: 1, html: true }); });




    // Center the graph
    var initialScale = 0.75;
    svg.call(zoom.transform, d3.zoomIdentity.translate(
        (svg.attr("width") - g.graph().width * initialScale) / 2, 20).scale(initialScale));

    svg.attr('height', g.graph().height * initialScale + 40);

    </script>
"""


start_node_idx = 0

node_count = len(model_proto.graph.node)
node_output_to_node_map = []
node_protos = []


def name_to_color(name: str):
    # hash the name to a color
    hash = 0
    for i in range(len(name)):
        hash = ord(name[i]) + ((hash << 5) - hash)
    color = "#"
    for i in range(3):
        value = (hash >> (i * 8)) & 0xFF
        color += f"{value:02x}"
    return color


def title_to_width_and_height(title: str):
    """According to the name length to calculate the width and height in pixel"""
    width = min(500, len(title) * 10)
    height = 40
    return width, height


subggraph_index = [0]
color_map = {}

normal_node_count = 0
subgraph_node_count = 0
# print(sorted_subgraph_starts)
while start_node_idx < node_count:

    if start_node_idx not in sorted_subgraph_starts:
        # print(f"normal handle start_node_idx: {start_node_idx}")
        cur_node = model_proto.graph.node[start_node_idx]
        assert (
            s[start_node_idx] != "|"
        ), f"Failure: found | at index {start_node_idx}: {s[start_node_idx]}"

        # if cur_node.op_type == "YieldOp":
        #     print(f"found YieldOp at {start_node_idx}, break")
        #     break

        normal_node_count += 1
        node_protos.append(cur_node)
        start_node_idx += 1
        continue

    # print(f"handle subgraph start_node_idx: {start_node_idx}")
    end_node_idx = start_node_idx + sorted_subgraph_starts[start_node_idx][0]

    subgraph_pattern = sorted_subgraph_starts[start_node_idx][1]
    subggraph_index[0] += 1

    # print(f"count: {subgraph_starts[start_node_idx][0]}, v_node_name: {v_node_name}")

    subgraph = model_proto.graph.node[start_node_idx:end_node_idx]
    subgraph_inputs, subgraph_outputs = get_external_inputs_and_outputs_for_subgraph(
        subgraph
    )
    # print(subgraph_inputs)
    # print(subgraph_outputs)

    # create a new ONNX node in 'custom' domain, opset 1, optype 'Subgraph',
    # taking subgraph_inputs as inputs and producing subgraph_outputs as outputs.
    # There is no attribute for the subgraph node, so we use an empty attribute list.
    subgraph_node = onnx.helper.make_node(
        "Subgraph",
        subgraph_inputs,
        subgraph_outputs,
        name="Subgraph" + str(subggraph_index[0]),
        doc_string=f"{subgraph_pattern}",
        domain="custom",
        opset_version=1,
        attributes=[],  # no attributes
    )
    node_protos.append(subgraph_node)
    # color_map[subgraph_node.name] = name_to_color(subgraph_pattern)
    # print(f"reduce number of nodes: {subgraph_starts[start_node_idx][0] - 1}")

    start_node_idx = end_node_idx

    subgraph_node_count += 1


node_output_to_node_map = {}
print(
    f"reduced node count to {len(node_protos)}, normal_node_count: {normal_node_count}, subgraph_node_count: {subgraph_node_count}"
)
for node in node_protos:
    for out in node.output:
        node_output_to_node_map[out] = node

# print(node_output_to_node_map)


def get_node_unique_id(node):
    if node.op_type == "Subgraph":
        return f"{_get_subgraph_str_from_encoded_str(node.doc_string)[:15]}_{node.name}"

    return node.name


# Open a file named gen.html in write mode
f = open("gen4.html", "w")
f.write(prefix + "\n")
f.write("var states = {};")
# loop through node_protos
for node in node_protos:
    color_str = ""
    description = ""
    # if node.name in color_map:
    if node.op_type == "Subgraph":
        color_str = f", style: 'fill: {name_to_color(node.doc_string)}'"

        description = (
            "Pattern graph: "
            + _get_subgraph_str_from_encoded_str(node.doc_string)
            + "<br>"
        )

        description += f"Frequency: {sorted_rets[node.doc_string].freq}<br>"

    title = get_node_unique_id(node)
    w, h = title_to_width_and_height(title)
    # node_str = f"g.setNode('{node.name}', {{ label: '{node.name}', width: {2* w}, height: {h}, labelStyle: 'font-size: 2em' {color_str}}});"
    append_state_str = f"states['{title}'] = {{ description: `{description}`, width: {w}, height: {h} {color_str}}};"
    f.write(append_state_str + "\n")


for node in node_protos:
    for inp in node.input:
        # Ignore the graph input
        if inp not in node_output_to_node_map:
            continue
        # print(
        #     f"edge: {node_output_to_node_map[inp].name} -> {node.name} --> {node.op_type}")
        edge_str = f"g.setEdge('{get_node_unique_id(node_output_to_node_map[inp])}', '{get_node_unique_id(node)}');"
        f.write(edge_str + "\n")

    # write the node to the file
    # f.write(str(node) + "\n")
f.write(postfix + "\n")

f.close()
