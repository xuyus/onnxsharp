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

import argparse
import onnx

from onnxsharp import (
    Model,
    clip_subgraph_around,
    generate_safe_file_name,
)


def cli_onnx_summarize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    # int argument level, by default being 0
    parser.add_argument("--level", type=int, default=0)
    # bool argument include_shape, by default being False
    parser.add_argument("--include_shape", type=bool, default=False)
    # bool argument summary_inputs, by default being False
    parser.add_argument("--include_inputs", type=bool, default=False)

    args = parser.parse_args()

    m = Model.load_model(args.model)

    print("=== Summarizing Nodes ===")
    m._graph.summarize_nodes(args.level, include_shape=args.include_shape)

    if args.include_inputs:
        print("=== Summarizing Inputs ===")
        m._graph.summarize_inputs()


def cli_onnx_clip_subgraph():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)

    # Add a argument that could be either node_name or node_output_name
    parser.add_argument("--node_name", type=str)
    parser.add_argument("--node_output_name", type=str)

    args = parser.parse_args()

    if args.node_name is None and args.node_output_name is None:
        raise ValueError("Either node_name or node_output_name must be provided.")

    if args.node_name is not None and args.node_output_name is not None:
        raise ValueError("Only one of node_name or node_output_name can be provided.")

    m = Model.load_model(args.model)
    if args.node_output_name:
        new_g = clip_subgraph_around(m._graph, args.node_output_name)
        new_m = Model.copy_config(m, new_g)
        new_m.save_model(f"{generate_safe_file_name(args.node_output_name)}.onnx")

    if args.node_name:
        # Get one of the output names of the node
        target_node = [None]

        def input_filter_func(node):
            if node.name == args.node_name:
                target_node[0] = node

        m._graph.iterate_node(input_filter_func)

        output_name = target_node[0].output_arg_names[0]
        new_g = clip_subgraph_around(m._graph, output_name)
        new_m = Model.copy_config(m, new_g)
        new_m.save_model(f"{generate_safe_file_name(args.node_name)}.onnx")


def cli_onnx_get_nodes():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    # Either op_type or node_name or node_output_name must be provided
    parser.add_argument("--op_type", type=str)
    parser.add_argument("--node_name", type=str)
    parser.add_argument("--node_output_name", type=str)

    args = parser.parse_args()

    if (
        args.op_type is None
        and args.node_name is None
        and args.node_output_name is None
    ):
        raise ValueError(
            "Either op_type or node_name or node_output_name must be provided."
        )

    if args.op_type is not None and args.node_name is not None:
        raise ValueError("Only one of op_type or node_name can be provided.")

    if args.op_type is not None and args.node_output_name is not None:
        raise ValueError("Only one of op_type or node_output_name can be provided.")

    if args.node_name is not None and args.node_output_name is not None:
        raise ValueError("Only one of node_name or node_output_name can be provided.")

    m = Model.load_model(args.model)

    nodes = []

    def output_filter_func(node):
        if args.op_type is not None and node.type == args.op_type:
            nodes.append(node)
        if args.node_name is not None and node.name == args.node_name:
            nodes.append(node)
        if (
            args.node_output_name is not None
            and args.node_output_name in node.output_arg_names
        ):
            nodes.append(node)

    m._graph.iterate_node(output_filter_func)

    for node in nodes:
        print(node)


def cli_onnx_to_text():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    model_proto = onnx.load(args.model)
    # onnx.save(model_proto, args.src + ".txt",format=Text)

    text_file = open(args.model + ".txt", "w")
    print(f"Text file written to {args.model + '.txt'}")
    text_file.write(str(model_proto))
    text_file.close()
