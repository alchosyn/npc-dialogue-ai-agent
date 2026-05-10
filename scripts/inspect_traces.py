"""Print the steps of the most recent trace under data/traces/."""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from npc_agent.config import TRACE_DIR  # noqa: E402


def main() -> None:
    files = sorted(glob.glob(str(TRACE_DIR / "**" / "*.json"), recursive=True))
    if not files:
        print("没有找到任何 trace 文件")
        return
    with open(files[-1], "r", encoding="utf-8") as f:
        trace = json.load(f)
    for step in trace["steps"]:
        print(step)


if __name__ == "__main__":
    main()
