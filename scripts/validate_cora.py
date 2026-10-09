"""Strict validation for the standard Cora files used by this project."""
from __future__ import annotations

import argparse
from pathlib import Path

EXPECTED_NODES = 2708
EXPECTED_FEATURES = 1433
EXPECTED_CLASSES = 7
EXPECTED_EDGES = 5429


def validate(data_dir: str) -> None:
    root = Path(data_dir)
    content = root / "cora.content"
    cites = root / "cora.cites"
    if not content.exists() or not cites.exists():
        raise FileNotFoundError(
            f"Expected {content} and {cites}. Run: python scripts/prepare_cora.py"
        )

    ids = set()
    feature_count = None
    labels = set()
    rows = 0

    with content.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 3:
                raise ValueError(f"Malformed cora.content line {line_no}")
            paper_id = parts[0]
            if paper_id in ids:
                raise ValueError(f"Duplicate paper id {paper_id!r} at line {line_no}")
            ids.add(paper_id)
            current_features = len(parts) - 2
            feature_count = current_features if feature_count is None else feature_count
            if current_features != feature_count:
                raise ValueError(
                    f"Inconsistent feature count at line {line_no}: "
                    f"expected {feature_count}, found {current_features}"
                )
            labels.add(parts[-1])
            rows += 1

    edge_count = 0
    with cites.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Malformed cora.cites line {line_no}: {line!r}")
            if parts[0] not in ids or parts[1] not in ids:
                raise ValueError(f"Dangling citation at line {line_no}: {parts!r}")
            edge_count += 1

    checks = {
        "nodes": (rows, EXPECTED_NODES),
        "features": (feature_count, EXPECTED_FEATURES),
        "classes": (len(labels), EXPECTED_CLASSES),
        "citation rows": (edge_count, EXPECTED_EDGES),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"Cora {name} mismatch: expected {expected}, found {actual}")

    print("Cora validation PASSED")
    print(f"  nodes:     {rows}")
    print(f"  features:  {feature_count}")
    print(f"  classes:   {len(labels)}")
    print(f"  citations: {edge_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/cora")
    args = parser.parse_args()
    validate(args.data_dir)
