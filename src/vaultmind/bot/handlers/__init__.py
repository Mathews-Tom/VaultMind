"""Handler package — delegates to individual handler modules.

Exports CommandHandlers as a facade preserving the original public interface.
"""

from __future__ import annotations

from vaultmind.bot.handlers.context import HandlerContext

__all__ = ["CommandHandlers", "HandlerContext"]

# Re-export CommandHandlers from the facade (commands.py)
# The actual class is defined in bot/commands.py which imports from here.
