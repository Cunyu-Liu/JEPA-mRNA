import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import train_plan_a
import xens2_eval
from structrfm_round48 import install_structrfm_training


if __name__ == "__main__":
    if "--pretrained-structrfm" not in sys.argv:
        raise SystemExit("--pretrained-structrfm is required")
    index = sys.argv.index("--pretrained-structrfm")
    if index + 1 >= len(sys.argv):
        raise SystemExit("--pretrained-structrfm requires a path")
    pretrained = sys.argv[index + 1]
    install_structrfm_training(train_plan_a, xens2_eval, pretrained)
    del sys.argv[index:index + 2]
    train_plan_a.main()
