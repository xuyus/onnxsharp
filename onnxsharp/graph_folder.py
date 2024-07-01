# Copyright 2024 XUYUS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from collections import OrderedDict
import copy
from typing import List, Set, Tuple
import onnx
from onnxsharp import Node


count_threshold = 2
cluster_threshold = 64


class Encoder:
    def __init__(self, node_list: List[Node]):
        self._op_type_to_int = []
        self._node_list = node_list
        for node in self._node_list:
            if node.type not in self._op_type_to_int:
                self._op_type_to_int.append(node.type)

    def get_alphabet_from_int(self, int_val: int):
        if int_val >= 0 and int_val <= 9:
            return chr(int_val + ord("0"))
        elif int_val >= 10 and int_val <= 35:
            return chr(int_val - 10 + ord("a"))
        elif int_val >= 36 and int_val <= 61:
            return chr(int_val - 36 + ord("A"))
        raise ValueError(f"unsupported int_val: {int_val}")

    def get_op_type_from_alphabet(self, val: str):
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

        return self._op_type_to_int[index]

    def encode(self) -> str:
        type_ints = []
        # loop through the model_proto.graph.node
        for node in self._node_list:
            ascii_int = self._op_type_to_int.index(node.type)
            type_ints.append(self.get_alphabet_from_int(ascii_int))

        return "".join(type_ints)

    def decode(self, s: str) -> List[str]:
        return [self.get_op_type_from_alphabet(c) for c in s]


class NodeInfo:
    def __init__(
        self,
        model_proto: List[Node],
        op_type,
        name,
        inputs: List[str],
        outputs: List[str],
        is_subgraph: bool,
        contained_graph_encoded: str,
        node_offset: int,
        depth: int,
        disable_sub_nodeinfo_detection: bool = False,
    ):
        self.op_type = op_type
        self.name = name
        self.inputs = copy.deepcopy(inputs)
        self.outputs = copy.deepcopy(outputs)
        self.is_subgraph = is_subgraph
        self.contained_graph_encoded = copy.deepcopy(contained_graph_encoded)
        self.node_offset = node_offset  # inclusive
        self.model_proto = model_proto
        self.depth = depth

        self.node_infos = []
        if not disable_sub_nodeinfo_detection and self.is_subgraph:
            self.node_infos = fold_graph(
                self.contained_graph_encoded,
                model_proto,
                self.node_offset,
                depth + 1,
            )

    def __str__(self):
        return f"NodeInfo: {self.op_type}, {self.name}, {self.inputs}, {self.outputs}, {self.is_subgraph}, {self.contained_graph_encoded}, {self.node_offset}"

    def __repr__(self):
        return self.__str__()


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


class Pattern:
    def __init__(self, freq, start_offsets, length):
        self.freq = freq
        self.start_offsets = copy.deepcopy(start_offsets)
        self.length = length


def get_external_inputs_and_outputs_for_subgraph(
    subgraph: List[onnx.NodeProto],
) -> Tuple[Set[str], Set[str]]:
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


def fold_graph(
    s: str, node_list: List[Node], node_offset: int, depth: int = 0
) -> List[NodeInfo]:

    print(f"handle sequence {s}, its offset in original str: {node_offset}")
    seq_len = len(s)

    # First round, find the most longest repeated subgraph pattern.
    repeated_pattern = {}
    for i in range(seq_len):
        for j in range(seq_len - 1, i, -1):
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
                break

    sorted_repeated_pattern = OrderedDict(
        sorted(
            repeated_pattern.items(),
            key=lambda item: item[1].freq,
            reverse=True,
        )
    )

    valid_combinations = {}
    for k, v in sorted_repeated_pattern.items():
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

    rets = {}
    subgraph_starts = {}
    for k, v in valid_combinations.items():
        if v.freq >= count_threshold:
            cur_substr = k
            s = s.replace(cur_substr, "|" * len(cur_substr))
            for x in v.start_offsets:
                subgraph_starts[x] = [v.length, cur_substr]

            rets[cur_substr] = Pattern(v.freq, v.start_offsets, v.length)
            print(
                f"apply valid_combinations: {cur_substr}, frequency: {v.freq}, length: {v.length}"
            )

    has_update = True

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
                    break
                j -= 1

            if has_update:
                break

            i += 1

    print(f"s: {s}")

    sorted_rets = rets  # OrderedDict(sorted(rets.items(), key=lambda item: item[0], reverse=True))

    for k, v in sorted_rets.items():
        subgraph_str = k  # _get_subgraph_str_from_encoded_str(k)
        # print(subgraph_str, ':', v.freq)
        for x in v.start_offsets:
            subgraph_starts[x] = [v.length, subgraph_str]

    sorted_subgraph_starts = OrderedDict(
        sorted(subgraph_starts.items(), key=lambda item: item[0], reverse=False)
    )

    start_node_idx = 0
    node_count = len(s)
    node_infos = []
    subggraph_index = [0]
    normal_node_count = 0
    subgraph_node_count = 0
    # print(sorted_subgraph_starts)
    while start_node_idx < node_count:

        if start_node_idx not in sorted_subgraph_starts:
            # print(f"normal handle start_node_idx: {start_node_idx}")
            cur_node = node_list[start_node_idx + node_offset]
            assert (
                start_node_idx < len(s) and s[start_node_idx] != "|"
            ), f"Failure: found | at index {start_node_idx}: {s[start_node_idx] if start_node_idx < len(s) else 'out of range'}, s: {s}"

            normal_node_count += 1
            node_infos.append(
                NodeInfo(
                    node_list,
                    cur_node.type,
                    cur_node.name,
                    cur_node.input_arg_names,
                    cur_node.output_arg_names,
                    False,
                    "",
                    start_node_idx + node_offset,
                    depth,  # depth
                )
            )
            start_node_idx += 1
            continue

        end_node_idx = start_node_idx + sorted_subgraph_starts[start_node_idx][0]
        subgraph_pattern = sorted_subgraph_starts[start_node_idx][1]
        subggraph_index[0] += 1

        # subgraph = model_proto.graph.node[start_node_idx + node_offset: end_node_idx + node_offset]
        # subgraph_inputs, subgraph_outputs = get_external_inputs_and_outputs_for_subgraph(subgraph)

        node_infos.append(
            NodeInfo(
                node_list,
                "Subgraph",
                "Subgraph" + str(subggraph_index[0]),
                [],
                [],
                True,
                subgraph_pattern,
                start_node_idx + node_offset,
                depth,  # depth
            )
        )

        start_node_idx = end_node_idx

        subgraph_node_count += 1

    print(
        f"reduced node count to {len(node_infos)}, normal_node_count: {normal_node_count}, subgraph_node_count: {subgraph_node_count}"
    )
    return node_infos
