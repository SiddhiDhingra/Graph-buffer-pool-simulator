from src.thgb_page_graph import build_thgb_page_graph

result = build_thgb_page_graph(page_size=50)

print("Nodes:", result["num_nodes"])
print("Pages:", result["num_pages"])
print("Original edge records:", result["edge_records"])
print("Edge counts by type:", result["edge_type_counts"])
print("Within-page edges by type:", result["self_page_edge_counts"])
print("Unique directed page connections:", result["num_unique_cross_page_connections"])
