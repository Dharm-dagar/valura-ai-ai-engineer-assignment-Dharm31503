"""Synchronous safety guard. No LLM. No network. Pure regex + literal matching."""
from .guard import SafetyGuard, SafetyVerdict, check, default_guard

__all__ = ["SafetyGuard", "SafetyVerdict", "check", "default_guard"]
