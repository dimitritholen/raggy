"""Factory for creating command instances."""

from .base import Command
from .commands import (
    BuildCommand,
    DiagnoseCommand,
    ForgetCommand,
    InitCommand,
    InteractiveCommand,
    OptimizeCommand,
    RecallCommand,
    RememberCommand,
    SearchCommand,
    StatusCommand,
    TestCommand,
    ValidateCommand,
)


class CommandFactory:
    """Factory for creating command instances."""

    _commands = {
        "init": InitCommand,
        "build": BuildCommand,
        "rebuild": BuildCommand,
        "search": SearchCommand,
        "interactive": InteractiveCommand,
        "status": StatusCommand,
        "optimize": OptimizeCommand,
        "test": TestCommand,
        "diagnose": DiagnoseCommand,
        "validate": ValidateCommand,
        "remember": RememberCommand,
        "recall": RecallCommand,
        "forget": ForgetCommand,
    }

    @classmethod
    def create_command(cls, command_name: str) -> Command:
        """Create a command instance.

        Args:
            command_name: Name of the command to create

        Returns:
            Command: Command instance

        Raises:
            ValueError: If command name is unknown
        """
        command_class = cls._commands.get(command_name)
        if command_class is None:
            raise ValueError(f"Unknown command: {command_name}")
        return command_class()
