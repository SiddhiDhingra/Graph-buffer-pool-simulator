from src.thgb_page_vectors import build_thgb_page_vectors

vectors = build_thgb_page_vectors(page_size=50)

print("Page vectors shape:", vectors.shape)
print("Data type:", vectors.dtype)
print("All values finite:", bool(__import__("numpy").isfinite(vectors).all()))
print("First vector norm:", float(__import__("numpy").linalg.norm(vectors[0])))
