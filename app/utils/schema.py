from __future__ import annotations
import yaml
from typing import Dict

def load_sink_contract(path: str) -> Dict[str, str]:
    y = yaml.safe_load(open(path, "r", encoding="utf-8"))
    return {k: v for k, v in y["sink_contract"]["columns"].items()}
