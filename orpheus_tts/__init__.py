"""Orpheus Text-to-Speech System."""

__version__ = "0.1.0"

# Import and expose the main function
# from .main import generate_tokens_sync
from .decoder import tokens_decoder_sync
from .engine_class import OrpheusModel