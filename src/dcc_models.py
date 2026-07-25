"""Compatibility layer for checkpoints saved before the PDCC-MER rename."""

from src.pdcc_models import *  # noqa: F401,F403

# Old full-module checkpoints resolve this exact module attribute while loading.
DCCModel = PDCCModel
