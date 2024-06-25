from typing import Dict
from model_explorer import Adapter, AdapterMetadata, ModelExplorerGraphs, graph_builder
import onnx
from onnxsharp.graph_folder import fold_graph, Encoder, NodeInfo


global_subgraph_index = [0]

class ONNXAdapter(Adapter):

  metadata = AdapterMetadata(id='onnxsharp',
                             name='ONNX adapter',
                             description='ONNX adapter!',
                             source_repo='https://github.com/xuyus/onnx-sharp',
                             fileExts=['onnx'])

  # Required.
  def __init__(self):
    super().__init__()
    self._already_imported = set()


  def handle_layer_recursively(self, namespace, current_recursive_depth, encoder, encoded_str, model_proto, node_output_to_node_map, start_index, graph):

    node_infos = fold_graph(encoder, encoded_str, model_proto, start_index)


    current_namespace = namespace
    

    # Loop the node_protos, if the op_type is not 'Subgraph', the namesapce is current_namespace, otherwise, we create a new namespace following current_namespace/new_namespace_name.
    for node_info in node_infos:
      ns = current_namespace
      if node_info.is_subgraph:
        global_subgraph_index[0] += 1
        self.handle_layer_recursively(current_namespace + ('/' if current_namespace else '') + f'pattern_{self.str_to_hashed_int(node_info.contained_graph_encoded)}_instance_' + str(global_subgraph_index[0]), 
                                      current_recursive_depth + 1, 
                                      encoder, 
                                      node_info.contained_graph_encoded, 
                                      model_proto, 
                                      node_output_to_node_map, 
                                      node_info.node_offset, 
                                      graph)
        continue

      node_id = f"{node_info.name}({node_info.op_type})"
      assert node_id not in self._already_imported, f"Node {node_id} already imported"
      new_node = graph_builder.GraphNode(id=node_id, label=node_id, namespace=ns)


      
      for inp in node_info.inputs:
        if inp in node_output_to_node_map: # skip the graph inputs.
          new_node.incomingEdges.append(graph_builder.IncomingEdge(sourceNodeId=f"{node_output_to_node_map[inp].name}({node_output_to_node_map[inp].op_type})"))
        else:
          # print("Warning: input not found: ", inp)
          new_node.incomingEdges.append(graph_builder.IncomingEdge(sourceNodeId=inp))
      graph.nodes.append(new_node)

  def str_to_hashed_int(self, s: str) -> int:
    return hash(s) % ((1 << 31) - 1)

  def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
    model_proto = onnx.load(model_path)
    encoder = Encoder(model_proto)
    encoded_str = encoder.encode()


    # Create a graph for my road trip.
    graph = graph_builder.Graph(id='onnx_model')

    node_output_to_node_map = {}
    for node in model_proto.graph.node:

      for output in node.output:
        node_output_to_node_map[output] = node
      

    # Create node for graph inputs, initializers and outputs.
    graph_inputs = []
    for input in model_proto.graph.input:
      new_node = graph_builder.GraphNode(id=input.name, label=input.name, namespace='')
      graph.nodes.append(new_node)
      graph_inputs.append(input.name)

    for initializer in model_proto.graph.initializer:
      if initializer.name in graph_inputs:
        continue
      new_node = graph_builder.GraphNode(id=initializer.name, label=initializer.name, namespace='')
      graph.nodes.append(new_node)


    self.handle_layer_recursively('', 0, encoder, encoded_str, model_proto, node_output_to_node_map, 0, graph)


    # Add incoming edge for output nodes for the graph.
    for output in model_proto.graph.output:
      assert output.name in node_output_to_node_map
      new_node = graph_builder.GraphNode(id=output.name, label=output.name, namespace='')
      graph.nodes.append(new_node)
      new_node.incomingEdges.append(graph_builder.IncomingEdge(sourceNodeId=f"{node_output_to_node_map[output.name].name}({node_output_to_node_map[output.name].op_type})"))


    return {'graphs': [graph]}