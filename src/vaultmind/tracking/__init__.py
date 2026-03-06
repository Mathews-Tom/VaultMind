"""Tracking — user interaction preferences and usage analytics."""

from vaultmind.tracking.analyzer import (
    PreferenceInsights,
    analyze_preferences,
    generate_preference_report,
)
from vaultmind.tracking.preferences import (
    Interaction,
    InteractionType,
    PreferenceStore,
)

__all__ = [
    "Interaction",
    "InteractionType",
    "PreferenceInsights",
    "PreferenceStore",
    "analyze_preferences",
    "generate_preference_report",
]
