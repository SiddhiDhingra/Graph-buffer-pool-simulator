"""Build logical pages and maintain Node->Page / Page->Nodes mappings."""
from __future__ import annotations

from typing import Dict, List, Optional

from src.page import Page

PAGE_SIZE = 50


class PageTable:
    def __init__(self, num_nodes: int, page_size: int = PAGE_SIZE):
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.num_nodes = num_nodes
        self.page_size = page_size
        self.node_to_page: Dict[int, int] = {}
        self.page_to_nodes: Dict[int, List[int]] = {}
        self.pages: Dict[int, Page] = {}

    def build_pages(self) -> "PageTable":
        """Page 0 -> nodes 0..49, Page 1 -> nodes 50..99, ..."""
        self.node_to_page.clear()
        self.page_to_nodes.clear()
        self.pages.clear()
        for start in range(0, self.num_nodes, self.page_size):
            pid = start // self.page_size
            nodes = list(range(start, min(start + self.page_size, self.num_nodes)))
            self.page_to_nodes[pid] = nodes
            self.pages[pid] = Page(pid, tuple(nodes))
            for n in nodes:
                self.node_to_page[n] = pid
        return self

    def get_page_for_node(self, node_id: int) -> int:
        if node_id not in self.node_to_page:
            raise KeyError(f"node {node_id} is not in the page table")
        return self.node_to_page[node_id]

    def get_nodes_in_page(self, page_id: int) -> List[int]:
        return self.page_to_nodes[page_id]

    @property
    def num_pages(self) -> int:
        return len(self.pages)


_default_table: Optional[PageTable] = None


def build_pages(graph_or_num_nodes, page_size: int = PAGE_SIZE) -> PageTable:
    """Build pages from a CoraGraph (or a node count). Also sets the module
    default table used by get_page_for_node()."""
    global _default_table
    n = graph_or_num_nodes if isinstance(graph_or_num_nodes, int) \
        else graph_or_num_nodes.num_nodes
    _default_table = PageTable(n, page_size).build_pages()
    return _default_table


def get_page_for_node(node_id: int, table: Optional[PageTable] = None) -> int:
    table = table or _default_table
    if table is None:
        raise RuntimeError("call build_pages() first")
    return table.get_page_for_node(node_id)
