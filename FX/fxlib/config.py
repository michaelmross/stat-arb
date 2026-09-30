"""Config loading and the config hash that proves parameters were frozen."""
from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: dict) -> str:
    """sha256 over the *parsed* config, so formatting/comments don't change it."""
    return hashlib.sha256(_canonical(cfg).encode()).hexdigest()[:16]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.toml"
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)
    if cfg["oanda"]["environment"] != "practice":
        raise SystemExit(
            f"Refusing to run: config.toml [oanda].environment is "
            f"{cfg['oanda']['environment']!r}. This experiment is paper-only; "
            f"only 'practice' is permitted."
        )
    cfg["_path"] = str(path)
    cfg["_hash"] = config_hash({k: v for k, v in cfg.items() if not k.startswith("_")})
    return cfg


def load_credentials(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "credentials.toml"
    if not Path(path).exists():
        raise SystemExit(
            f"Missing {path}.\n"
            f"Copy credentials.example.toml -> credentials.toml and fill in your\n"
            f"OANDA *practice* API token and account id\n"
            f"(Account Management Portal -> Manage API Access)."
        )
    with open(path, "rb") as fh:
        cred = tomllib.load(fh)["oanda"]
    tok, acct = cred.get("api_token", ""), cred.get("account_id", "")
    if not tok or tok.startswith("PUT-YOUR"):
        raise SystemExit(f"{path}: api_token is not filled in.")
    if not acct or acct.startswith("101-001-0000000"):
        raise SystemExit(f"{path}: account_id is not filled in.")
    return {"api_token": tok, "account_id": acct}


def apply_account_override(cfg: dict, cred: dict) -> dict:
    """Let a config pin itself to a specific account.

    Phase 1b runs two extra triangles, and OANDA may drop an existing pricing
    stream when a new one opens on the same account. Putting each triangle on its
    own account avoids the question rather than betting on it. credentials.toml
    still holds the single token -- both accounts are under it -- and this only
    redirects which account the stream and REST calls address.
    """
    override = (cfg.get("oanda") or {}).get("account_id")
    if override:
        cred = dict(cred)
        cred["account_id"] = override
    return cred


def instruments(cfg: dict) -> tuple[str, str, str]:
    o = cfg["oanda"]
    return o["instrument_a"], o["instrument_b"], o["instrument_c"]
