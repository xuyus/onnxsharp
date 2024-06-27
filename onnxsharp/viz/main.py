from typing import Dict, List
from model_explorer import Adapter, AdapterMetadata, ModelExplorerGraphs, graph_builder

from onnxsharp.graph_folder import fold_graph, Encoder, NodeInfo
from onnxsharp import Model, memory_efficient_topological_sort, Node


class ONNXAdapter(Adapter):

    metadata = AdapterMetadata(
        id="onnxsharp",
        name="ONNX adapter",
        description="ONNX adapter!",
        source_repo="https://github.com/xuyus/onnx-sharp",
        fileExts=["onnx"],
    )

    def __init__(self):
        super().__init__()
        self._already_imported = set()
        self._pattern_hash_count = {}

    def _str_to_hash_to_int(self, s: str) -> int:
        return hash(s) % ((1 << 31) - 1)

    def _get_namespace_from_encoded_str(self, current_ns, encoded_str: str) -> str:
        hashed_int = self._str_to_hash_to_int(encoded_str)
        if hashed_int not in self._pattern_hash_count:
            self._pattern_hash_count[hashed_int] = 0

        self._pattern_hash_count[hashed_int] += 1
        next_level_ns = (
            f"pattern_{hashed_int}_apperance_{self._pattern_hash_count[hashed_int]}"
        )

        if current_ns != "":
            return "/".join([current_ns, next_level_ns])
        return next_level_ns

    def node_info_to_node(
        self,
        node_info: NodeInfo,
        current_ns: str,
        node_output_to_node_map,
        graph: graph_builder.Graph,
    ):
        if node_info.is_subgraph:
            node_info_ns = self._get_namespace_from_encoded_str(
                current_ns, node_info.contained_graph_encoded
            )

            for sub_node_info in node_info.node_infos:
                self.node_info_to_node(
                    sub_node_info, node_info_ns, node_output_to_node_map, graph
                )
            return

        node_id = f"{node_info.name}({node_info.op_type})"
        new_node = graph_builder.GraphNode(
            id=node_id, label=node_id, namespace=current_ns
        )

        # print(f"Adding node {node_id} with {current_ns} namespace")

        for inp in node_info.inputs:
            if inp in node_output_to_node_map:  # skip the graph inputs.
                new_node.incomingEdges.append(
                    graph_builder.IncomingEdge(
                        sourceNodeId=f"{node_output_to_node_map[inp].name}({node_output_to_node_map[inp].type})"
                    )
                )
            else:
                new_node.incomingEdges.append(
                    graph_builder.IncomingEdge(sourceNodeId=inp)
                )
        graph.nodes.append(new_node)

    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        # model_proto = onnx.load(model_path)
        m = Model.load_model(model_path)
        node_orders: List[Node] = []
        memory_efficient_topological_sort(m._graph, node_orders)

        encoder = Encoder(node_orders)
        encoded_str = encoder.encode()

        # Create a graph for my road trip.
        graph = graph_builder.Graph(id="onnx_model")

        node_output_to_node_map = {}
        for node in node_orders:

            for output in node.output_arg_names:
                node_output_to_node_map[output] = node

        # Create node for graph inputs, initializers and outputs.
        graph_inputs = []
        for input in m._graph.input_names:
            new_node = graph_builder.GraphNode(id=input, label=input, namespace="")
            graph.nodes.append(new_node)
            graph_inputs.append(input)

        for initializer in m._graph.initializer_names:
            if initializer in graph_inputs:
                continue
            new_node = graph_builder.GraphNode(
                id=initializer, label=initializer, namespace=""
            )
            graph.nodes.append(new_node)

        # self.handle_layer_recursively(
        #     "", 0, encoder, encoded_str, model_proto, node_output_to_node_map, 0, graph
        # )
        node_infos = fold_graph(encoded_str, node_orders, 0, 0)

        if len(node_infos) > 1:
            print(f"Found {len(node_infos)} subgraphs")
            # Check if any of the node_info is contigous, if so we add a new NodeInfo and put them into the same node_info.
            merged_node_infos = []
            i = 0
            while i < len(node_infos):
                node_info = node_infos[i]
                if not node_info.is_subgraph:
                    merged_node_infos.append(node_info)
                    i += 1
                    continue

                contiguous_node_infos = [node_info]
                while (
                    i + 1 < len(node_infos)
                    and node_infos[i + 1].is_subgraph
                    and node_infos[i + 1].node_offset
                    == node_infos[i].node_offset
                    + len(node_infos[i].contained_graph_encoded)
                ):
                    i += 1
                    contiguous_node_infos.append(node_infos[i])

                if len(contiguous_node_infos) > 1:
                    print(
                        f"Found {len(contiguous_node_infos)} contiguous subgraphs: {contiguous_node_infos[0].node_offset} - {contiguous_node_infos[-1].node_offset}"
                    )
                    merged_node_info = NodeInfo(
                        contiguous_node_infos[0].model_proto,
                        "ContiguousSubgraph",
                        "ContiguousSubgraph",
                        [],
                        [],
                        True,
                        "".join(
                            [ni.contained_graph_encoded for ni in contiguous_node_infos]
                        ),
                        contiguous_node_infos[0].node_offset,
                        contiguous_node_infos[0].depth,
                        disable_sub_nodeinfo_detection=True,
                    )
                    merged_node_info.node_infos = contiguous_node_infos
                    merged_node_infos.append(merged_node_info)

                else:
                    merged_node_infos.append(node_info)

                i += 1

            node_infos = merged_node_infos

        for node_info in node_infos:
            self.node_info_to_node(node_info, "", node_output_to_node_map, graph)

        # Add incoming edge for output nodes for the graph.
        for output in m._graph.output_names:
            assert output in node_output_to_node_map
            new_node = graph_builder.GraphNode(id=output, label=output, namespace="")
            graph.nodes.append(new_node)
            new_node.incomingEdges.append(
                graph_builder.IncomingEdge(
                    sourceNodeId=f"{node_output_to_node_map[output].name}({node_output_to_node_map[output].type})"
                )
            )

        return {"graphs": [graph]}
