"""Dataset entry points. THGB DBLP-TA is the sole supported dataset."""
from src.thgb_loader import THGBGraph, load_thgb_dblp


def load_dataset(data_dir: str = "data/thgb/DBLP-TA", **kwargs) -> THGBGraph:
    return load_thgb_dblp(data_dir=data_dir, **kwargs)


__all__ = ["THGBGraph", "load_thgb_dblp", "load_dataset"]
