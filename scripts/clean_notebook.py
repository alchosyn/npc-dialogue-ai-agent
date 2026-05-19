"""Strip the broken `widgets` metadata from NPC_agent.ipynb."""
from pathlib import Path

import nbformat


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "notebooks" / "NPC_agent.ipynb"
    nb = nbformat.read(path, as_version=4)
    if "widgets" in nb.metadata:
        del nb.metadata["widgets"]
    nbformat.write(nb, path)
    print("清理完成")


if __name__ == "__main__":
    main()
