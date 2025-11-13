"""Base command interface for CLI."""

from typing import Any, Optional


class Command:
    """Base command interface."""

    def execute(self, args: Any, rag: Optional[Any] = None) -> None:
        """Execute the command.

        Args:
            args: Command line arguments
            rag: UniversalRAG instance (optional for some commands)
        """
        raise NotImplementedError