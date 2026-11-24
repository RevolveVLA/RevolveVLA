"""Test policies of the evaluation registry (policy plugins `frozen`, `noisy_expert`): bad policies that force the live recorder's
triggers. They replay the oracle's bag with the actions altered, so they run only under a live recorder (core/live.py)."""
from . import frozen, noisy_expert  # noqa: F401  (registers them)
