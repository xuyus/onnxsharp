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

    # inputs_of_classifier = set(["input1", "input", "output-0_grad"])
    inputs_of_classifier = set()

    def classifier_func(node: Node):
        input_arg_names = node.input_arg_names
        if (
            node.type == "Gemm"
            and len(input_arg_names) > 2
            and "_original_module.classifier" in input_arg_names[1]
            and input_arg_names[1].endswith(".weight")
            and "_original_module.classifier" in input_arg_names[2]
            and input_arg_names[2].endswith(".bias")
        ):
            for i in input_arg_names:
                inputs_of_classifier.add(i)

    print(inputs_of_classifier)
    m._graph.iterate_node(classifier_func)

    subgraph_info = Graph.LogicalSubgraphInfo(
        inputs_of_yield,
        list(inputs_of_classifier),
    )

    subgraph = Graph.from_logical_subgraph(m._graph, subgraph_info)
    new_m = Model.copy_config(m, subgraph)
    onnx.save(new_m.to_proto(), "pengwa_new_06_15.onnx")


if __name__ == "__main__":
    main()
