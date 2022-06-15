import onnx
import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args)
    model_proto = onnx.load(args.src)

    from onnxsharp import Model, Graph, Node

    m = Model.from_proto(model_proto)
    inputs_of_yield = []

    def func(node: Node):
        if node.type == "YieldOp":
            for i in node.input_arg_names:
                if i not in inputs_of_yield:
                    inputs_of_yield.append(i)

    print(inputs_of_yield)
    m._graph.iterate_node(func)
    subgraph_info = Graph.LogicalSubgraphInfo(
        inputs_of_yield,
        [],
    )

    subgraph = Graph.from_logical_subgraph(m._graph, subgraph_info)
    new_m = Model.copy_config(m, subgraph)
    onnx.save(new_m.to_proto(), "pengwa_new_06_15.onnx")


if __name__ == "__main__":
    main()
