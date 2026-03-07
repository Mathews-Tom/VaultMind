"""Tests for the error hierarchy."""

from pathlib import Path

from vaultmind.errors import VaultMindError
from vaultmind.llm.client import LLMError
from vaultmind.vault.security import PathTraversalError


def test_vaultmind_error_is_base():
    assert issubclass(LLMError, VaultMindError)
    assert issubclass(PathTraversalError, VaultMindError)


def test_llm_error_preserves_provider():
    err = LLMError("fail", provider="openai")
    assert err.provider == "openai"
    assert str(err) == "fail"
    assert isinstance(err, VaultMindError)


def test_path_traversal_preserves_fields():
    err = PathTraversalError("/etc/passwd", Path("/vault"))
    assert err.user_path == "/etc/passwd"
    assert isinstance(err, VaultMindError)


def test_catch_all_vaultmind_errors():
    """Verify that catching VaultMindError catches both subclasses."""
    try:
        raise LLMError("test", provider="anthropic")
    except VaultMindError:
        pass  # Should be caught

    try:
        raise PathTraversalError("../x", Path("/v"))
    except VaultMindError:
        pass  # Should be caught
