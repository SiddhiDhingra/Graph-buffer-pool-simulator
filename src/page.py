"""Logical database page: a fixed-size group of graph nodes."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    page_id: int
    node_ids: tuple          # node ids stored on this page

    @property
    def size(self) -> int:
        return len(self.node_ids)

    def contains(self, node_id: int) -> bool:
        return self.node_ids[0] <= node_id <= self.node_ids[-1]

    def __repr__(self) -> str:
        return (f"Page({self.page_id}: nodes {self.node_ids[0]}"
                f"-{self.node_ids[-1]}, size={self.size})")
