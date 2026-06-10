"""CLI utilities for qmri.

This module provides utility functions for the qmri command-line interface,
including progress bars, error handling, and file validation.
"""

from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

__all__ = [
    "create_progress",
    "error_handler",
    "validate_input_file",
    "validate_output_path",
]

# Shared console instance for consistent output
console = Console()
error_console = Console(stderr=True)

F = TypeVar("F", bound=Callable[..., Any])


def create_progress(description: str = "Processing") -> Progress:
    """Create a rich progress bar with standard styling.

    Args:
        description: Description to show alongside the progress bar.

    Returns:
        Configured rich Progress instance.

    Example:
        ```python
        with create_progress("Fitting ADC") as progress:
            task = progress.add_task("Processing...", total=100)
            for i in range(100):
                # do work
                progress.update(task, advance=1)
        ```
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def error_handler(func: F) -> F:
    """Wrap CLI commands with consistent error handling.

    Catches exceptions and displays them using rich formatting.
    Converts exceptions to click ClickException for proper exit codes.

    Args:
        func: The function to wrap.

    Returns:
        Wrapped function with error handling.
    """
    import click

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            error_console.print(f"[bold red]Error:[/] File not found: {e.filename}")
            raise click.ClickException(str(e)) from e
        except ValueError as e:
            error_console.print(f"[bold red]Error:[/] Invalid value: {e}")
            raise click.ClickException(str(e)) from e
        except click.ClickException:
            raise
        except Exception as e:
            error_console.print(f"[bold red]Error:[/] {type(e).__name__}: {e}")
            raise click.ClickException(str(e)) from e

    return wrapper  # type: ignore[return-value]


def validate_input_file(
    path: str | Path, extensions: tuple[str, ...] | None = None
) -> Path:
    """Validate that an input file exists and has correct extension.

    Args:
        path: Path to the input file.
        extensions: Allowed file extensions (e.g., ('.nii', '.nii.gz')).
            If None, any extension is allowed.

    Returns:
        Validated Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not in the allowed list.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(path)

    if extensions is not None:
        # Handle compound extensions like .nii.gz
        suffixes = "".join(file_path.suffixes).lower()
        if not any(suffixes.endswith(ext.lower()) for ext in extensions):
            msg = f"File must have extension {extensions}, got '{suffixes}'"
            raise ValueError(msg)

    return file_path


def validate_output_path(path: str | Path, create_parents: bool = True) -> Path:
    """Validate and prepare an output path.

    Args:
        path: Path for the output file.
        create_parents: Whether to create parent directories if they don't
            exist (default True).

    Returns:
        Validated Path object.

    Raises:
        ValueError: If the parent directory cannot be created or doesn't exist.
    """
    output_path = Path(path)

    parent = output_path.parent
    if not parent.exists():
        if create_parents:
            parent.mkdir(parents=True, exist_ok=True)
        else:
            msg = f"Output directory does not exist: {parent}"
            raise ValueError(msg)

    return output_path


def parse_values_file(path: str | Path) -> list[float]:
    """Parse a file containing numeric values.

    Supports files with values separated by whitespace, commas, or newlines.

    Args:
        path: Path to the values file.

    Returns:
        Parsed numeric values.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains non-numeric values.
    """
    file_path = validate_input_file(path)
    content = file_path.read_text()

    # Replace common separators with spaces
    content = content.replace(",", " ").replace("\n", " ").replace("\t", " ")

    values: list[float] = []
    for token in content.split():
        token = token.strip()
        if token:
            try:
                values.append(float(token))
            except ValueError as e:
                msg = f"Invalid numeric value: '{token}'"
                raise ValueError(msg) from e

    return values


def parse_values_string(values_str: str) -> list[float]:
    """Parse a string containing numeric values.

    Supports values separated by commas, spaces, or semicolons.

    Args:
        values_str: String containing numeric values.

    Returns:
        Parsed numeric values.

    Raises:
        ValueError: If the string contains non-numeric values.
    """
    # Replace common separators with spaces
    content = values_str.replace(",", " ").replace(";", " ")

    values: list[float] = []
    for token in content.split():
        token = token.strip()
        if token:
            try:
                values.append(float(token))
            except ValueError as e:
                msg = f"Invalid numeric value: '{token}'"
                raise ValueError(msg) from e

    return values
