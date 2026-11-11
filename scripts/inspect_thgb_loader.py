from collections import Counter
from src.thgb_loader import load_thgb_dblp

graph = load_thgb_dblp()

print("Nodes:", graph.num_nodes)
print("Node types:", dict(Counter(graph.node_types.tolist())))
print("Edges:", len(graph.edges))
print("Embedding matrix loaded:", graph.embeddings is not None)
print("First edge:", graph.edges[0])
print("First node text:", graph.texts[0][:100])
