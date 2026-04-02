#!/usr/bin/env python3
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def main() -> None:
    console = Console()
    title = Text("YOLOv13 Custom Kaggle DDP", style="bold bright_magenta")
    subtitle = Text("Welcome Eng.Ahmed ElBamby", style="bold purple")
    body = Text()
    body.append("Environment: ", style="bold white")
    body.append("/kaggle/work_here", style="bold orchid")
    body.append("  ->  ", style="bold #b085ff")
    body.append("Outputs: /kaggle/working", style="bold violet")

    panel = Panel(
        body,
        title=title,
        subtitle=subtitle,
        border_style="magenta",
        padding=(1, 2),
    )
    console.print(panel)


if __name__ == "__main__":
    main()
