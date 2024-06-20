import onnx

# Open a file named gen.html in write mode
f = open("gen.html", "w")


# load the model using onnx.load
# model_proto = onnx.load(
#     "C:\\Users\\pengwa\\dev\\onnx-sharp\\test\\clipped_subgraph.onnx"
# )
model_proto = onnx.load(
    R"C:\Users\pengwa\models\mistral_bingads\0522\0523_exp_ort_flash_attention_2_2048_run_001_execution_model_training.onnx"
)

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

  <script src="https://dagrejs.github.io/project/dagre/v0.7.5/dagre.js"></script>
  <script src="https://dagrejs.github.io/project/dagre-d3/latest/dagre-d3.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/d3@5"></script>



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
svg.call(zoom.transform, d3.zoomIdentity.translate((svg.attr("width") - g.graph().width * initialScale) / 2, 20).scale(initialScale));

svg.attr('height', g.graph().height * initialScale + 40);
console.log('end');
end = Date.now()
const millis3 = end - after;
console.log(`end elapsed = ${Math.floor(millis3 / 1000)}`);
</script>


"""

node_output_to_node_map = {}
for node in model_proto.graph.node:
    for output in node.output:
        node_output_to_node_map[output] = node

f.write(prefix + "\n")
# loop through the model_proto.graph.node
for node in model_proto.graph.node:
    node_str = f"g.setNode('{node.name}', {{ label: '{node.name}', width: 144, height: 100 }});"
    f.write(node_str + "\n")
    for inp in node.input:
        # Ignore the graph input
        if inp not in node_output_to_node_map:
            continue
        edge_str = f"g.setEdge('{node_output_to_node_map[inp].name}', '{node.name}');"
        f.write(edge_str + "\n")

    # write the node to the file
    # f.write(str(node) + "\n")
f.write(postfix + "\n")

f.close()

# g.setNode("kspacey",    { label: "Kevin Spacey",  width: 144, height: 100 });
# g.setNode("swilliams",  { label: "Saul Williams", width: 160, height: 100 });
# g.setNode("bpitt",      { label: "Brad Pitt",     width: 108, height: 100 });
# g.setNode("hford",      { label: "Harrison Ford", width: 168, height: 100 });
# g.setNode("lwilson",    { label: "Luke Wilson",   width: 144, height: 100 });
# g.setNode("kbacon",     { label: "Kevin Bacon",   width: 121, height: 100 });

# // Add edges to the graph.
# g.setEdge("kspacey",   "swilliams");
# g.setEdge("swilliams", "kbacon");
# g.setEdge("bpitt",     "kbacon");
# g.setEdge("hford",     "lwilson");
# g.setEdge("lwilson",   "kbacon");
