import os
from pathlib import Path

_ENV_LOADED = False


def _load_repo_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    project_root = Path(__file__).resolve().parents[4]
    demo_root = Path(__file__).resolve().parents[3]
    candidate_env_files = [
        project_root / ".env",
        demo_root / ".env",
    ]

    for env_path in candidate_env_files:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)

    _ENV_LOADED = True
