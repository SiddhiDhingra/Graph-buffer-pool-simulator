from src.thgb_graph_adapter import build_thgb_page_graph

graph = build_thgb_page_graph(page_size=50)

print("PageGraph class:", type(graph).__name__)
print("Pages with cross-page connections:", len(graph.weights))
print("Internal edge counts:", sum(graph.internal.values()))
print("Directed weighted entries:", sum(len(v) for v in graph.weights.values()))

sample_page = 0
print("Sample neighbors of page 0:", graph.neighbors(sample_page)[:10])
