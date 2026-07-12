"""Local REST and MCP service for corpus search and synchronization."""

from .api import create_app
from .core import CorpusService

__all__ = ["CorpusService", "create_app"]
