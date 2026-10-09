"""Download and install the standard Cora citation dataset.

This script is intentionally explicit: it downloads only the standard Cora
archive, verifies that cora.content and cora.cites are present, and installs
those two files under data/cora/.
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

DEFAULT_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"


def download(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "GraphBufferPoolResearch/1.0"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def install_cora(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        required = ("cora.content", "cora.cites")
        missing = [name for name in required if name not in members]
        if missing:
            raise RuntimeError(
                "The downloaded archive does not contain the expected Cora files: "
                + ", ".join(missing)
            )
        for name in required:
            src = tar.extractfile(members[name])
            if src is None:
                raise RuntimeError(f"Could not extract {name}")
            with (target / name).open("wb") as out:
                shutil.copyfileobj(src, out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the standard Cora dataset")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--target", default="data/cora")
    args = parser.parse_args()

    target = Path(args.target)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "cora.tgz"
        print(f"Downloading Cora from: {args.url}")
        download(args.url, archive)
        print("Installing cora.content and cora.cites...")
        install_cora(archive, target)

    print(f"Cora installed in: {target.resolve()}")
    print("Next: python check_cora.py")


if __name__ == "__main__":
    main()
