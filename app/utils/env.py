from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml, os
from dotenv import load_dotenv
from typing import Any
import re

class Env(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_config_env(config_path: str) -> dict[str, Any]:
    load_dotenv(override=False)
    cfg = load_yaml(config_path)
    def subst(val):
        if isinstance(val, str):
            for key, default in re.findall(r"\$\{env:([^,}]+)(?:,([^}]+))?\}", val):
                rep = os.getenv(key, default if default is not None else "")
                val = val.replace(f"${{env:{key}{','+default if default else ''}}}", rep)
        return val
    def walk(x):
        if isinstance(x, dict): return {k: walk(subst(v)) for k,v in x.items()}
        if isinstance(x, list): return [walk(v) for v in x]
        return subst(x)
    return walk(cfg)
