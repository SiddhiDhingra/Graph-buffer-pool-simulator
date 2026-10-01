from src.data_loader import load_cora
from src.page_builder import build_pages
from src.graph_locality import build_page_graph, calculate_graph_locality, top_local_pages
from src.workload_generator import WORKLOADS, to_page_requests

g = load_cora("data/cora")
print("nodes:", g.num_nodes)
print("feature shape:", g.features.shape)
print("classes:", len(g.label_names))
print("unique edges:", len(g.edges))

t = build_pages(g)
print("pages:", t.num_pages)
print("last page size:", len(t.page_to_nodes[t.num_pages - 1]))
print("node 10 is on page:", t.get_page_for_node(10))

pg = build_page_graph(g, t)
print("locality page 0 vs 1:", round(calculate_graph_locality(0, 1, pg), 4))
print("top related pages to page 0:", top_local_pages(0, pg, k=3))

print("\nDistinct pages touched in 1000 requests:")
for name, fn in WORKLOADS.items():
    nodes = fn(g, 1000, seed=1)
    pages = to_page_requests(nodes, t)
    print(f"  {name:10s} {len(set(pages))} of {t.num_pages}")