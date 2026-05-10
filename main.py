import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from npc_agent import run_chat_loop  # noqa: E402


if __name__ == "__main__":
    run_chat_loop()
