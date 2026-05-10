"""Delete the saved chat_history.json so the next run starts fresh."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from npc_agent.config import HISTORY_FILE  # noqa: E402


def main() -> None:
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()
        print("已清除旧存档")
    else:
        print("没有旧存档需要清除")


if __name__ == "__main__":
    main()
