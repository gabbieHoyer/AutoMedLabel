# clinical_utility/__init__.py
# Expose the triage package so callers can do:
#   from clinical_utility.triage import ...
from . import triage

__all__ = ["triage"]
