from __future__ import annotations

import os
from typing import Optional


def get_env_api_key(provider: str) -> Optional[str]:
    env_var = f"{provider.upper().replace('-', '_')}_API_KEY"
    return os.environ.get(env_var)


__all__ = ["get_env_api_key"]
