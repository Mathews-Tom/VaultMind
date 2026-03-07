"""MCP profile loading and validation."""

from __future__ import annotations

import logging
from typing import Any

from vaultmind.mcp.auth import ProfilePolicy

logger = logging.getLogger(__name__)

# Built-in default profiles (used when no profiles in config)
DEFAULT_PROFILES: dict[str, dict[str, Any]] = {
    "researcher": {
        "description": "Read-only access for research and Q&A tasks",
        "allowed_tools": ["vault_search", "vault_read", "vault_list", "graph_query", "graph_path"],
        "folder_scope": ["*"],
        "write_enabled": False,
    },
    "planner": {
        "description": "Read/write access for project planning",
        "allowed_tools": [
            "vault_search",
            "vault_read",
            "vault_write",
            "vault_list",
            "graph_query",
            "graph_path",
            "capture",
        ],
        "folder_scope": ["02-projects", "00-inbox"],
        "write_enabled": True,
        "max_note_size_bytes": 50000,
    },
    "full": {
        "description": "Unrestricted access",
        "allowed_tools": ["*"],
        "folder_scope": ["*"],
        "write_enabled": True,
        "requires_confirmation": True,
    },
}


def load_profile(name: str, config_profiles: dict[str, Any] | None = None) -> ProfilePolicy:
    """Load a profile by name from config or built-in defaults.

    Args:
        name: Profile name (e.g., "researcher", "planner", "full").
        config_profiles: Profile definitions from config/default.toml [mcp.profiles.*].

    Returns:
        ProfilePolicy for the requested profile.

    Raises:
        ValueError: If profile name is not found.
    """
    # Check config-defined profiles first
    if config_profiles and name in config_profiles:
        profile_data = config_profiles[name]
    elif name in DEFAULT_PROFILES:
        profile_data = DEFAULT_PROFILES[name]
    else:
        available = set()
        if config_profiles:
            available.update(config_profiles.keys())
        available.update(DEFAULT_PROFILES.keys())
        raise ValueError(f"Unknown profile '{name}'. Available: {', '.join(sorted(available))}")

    return ProfilePolicy(
        name=name,
        description=profile_data.get("description", ""),
        allowed_tools=frozenset(profile_data.get("allowed_tools", [])),
        folder_scope=tuple(profile_data.get("folder_scope", ["*"])),
        write_enabled=profile_data.get("write_enabled", False),
        max_note_size_bytes=profile_data.get("max_note_size_bytes"),
        requires_confirmation=profile_data.get("requires_confirmation", False),
    )
