"""Create the UTF-8 training-data copy used by the opt-in browser workbench."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "yc.csv"
DESTINATION = ROOT / "docs" / "assets" / "optimization-training-data.csv"


def export() -> Path:
    data = pd.read_csv(SOURCE, encoding="gb18030")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(DESTINATION, index=False, encoding="utf-8")
    return DESTINATION


if __name__ == "__main__":
    print(export())
