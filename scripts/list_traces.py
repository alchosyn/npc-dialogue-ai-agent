"""Dump the contents of every trace file under data/traces/."""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from npc_agent.config import TRACE_DIR  # noqa: E402


def main() -> None:
    files = glob.glob(str(TRACE_DIR / "**" / "*.json"), recursive=True)
    print(f"找到 {len(files)} 个 trace 文件")
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            print(json.dumps(json.load(fh), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
