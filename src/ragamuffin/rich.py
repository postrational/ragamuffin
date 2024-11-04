from collections.abc import Iterable
from typing import TypeVar

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

T = TypeVar("T")


def track(items: Iterable[T], total: int | None = None, description: str = "") -> Iterable[T]:
    """Track the progress of an iterable with a progress bar."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(description, total=total or len(items))
        for item in items:
            yield item
            progress.update(task, advance=1)


def format_list(items: list[str], numbered: bool = False) -> str:
    """Display a list of items."""
    formatted_string = ""
    for i, item in enumerate(items, start=1):
        if numbered:
            formatted_string += f"{i}. [cyan]{item}[/]\n"
        else:
            formatted_string += f"• [cyan]{item}[/]\n"

    return formatted_string
