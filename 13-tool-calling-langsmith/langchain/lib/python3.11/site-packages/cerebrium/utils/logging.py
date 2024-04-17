import logging
import os
import re
import sys
from typing import Literal, Optional, Union, Dict, Iterable

from rich.console import Console
from rich.logging import RichHandler
from termcolor import colored
from termcolor._types import Color, Attribute
from yaspin.core import Yaspin  # type: ignore

from cerebrium import datatypes

# Create a console object
console = Console(highlight=False)


__LOG_DEBUG_DELIMITERS__ = ["|| DEBUG ||", "|| END DEBUG ||"]
__LOG_INFO_DELIMITERS__ = ["|| INFO ||", "|| END INFO ||"]
__LOG_ERROR_DELIMITERS__ = ["|| ERROR ||", "|| END ERROR ||"]

__re_debug__ = re.compile(r"^\|\| DEBUG \|\| (.*) \|\| END DEBUG \|\|")
__re_info__ = re.compile(r"^\|\| INFO \|\| (.*) \|\| END INFO \|\|")
__re_error__ = re.compile(r"^\|\| ERROR \|\| (.*) \|\| END ERROR \|\|")


logger: Optional[logging.Logger] = None
rich_handler = RichHandler(
    console=console,
    rich_tracebacks=False,
    show_path=False,
    show_level=False,
    show_time=True if datatypes.task != "serve" else False,
    markup=True,
    highlighter=None,  # Explicitly disable syntax highlighting for log messages
)


def get_logger():
    global logger
    global rich_handler

    if logger is None:
        # re-initialise if the task is different
        rich_handler = RichHandler(
            console=console,
            rich_tracebacks=False,
            show_path=False,
            show_level=False,
            show_time=True,
            markup=True,
            highlighter=None,  # Explicitly disable syntax highlighting for log messages
        )
        # Setup logging globally
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S.%f",  # Include timestamp with milliseconds
            handlers=[rich_handler],
        )

        logger = logging.getLogger("rich")

    return logger


error_messages = {
    "disk quota exceeded": "💾 You've run out of space in your /persistent-storage. \n"
    "You can add more by running the command: `cerebrium storage increase-capacity <the_amount_in_GB>`"
}  # Error messages to check for


def log_formatted_response(log_line: str, use_console: bool = False):
    ##This function removes timestamps and prints based on the color passed (ie: INFO, ERROR etc)

    global logger
    if logger is None:
        logger = get_logger()
    pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{1,9}Z(?: \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}: WARNING\/MainProcess\])?|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{1,9}Z \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+"
    log_line = re.sub(pattern, "", log_line)
    log_line = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+", "", log_line)
    log_line = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z", "", log_line)

    if not log_line.strip():
        return  # Skip logging if the line is empty or whitespace

    # Check if the log line contains "disk quota exceeded"
    if "disk quota exceeded" in log_line.lower():
        if error_msg := error_messages.get("disk quota exceeded"):
            # Format and print the error message
            formatted_msg = f"\n🚨 Build failed! \n" f"{error_msg}"
            logger.error(f"[red]{formatted_msg}[/red]")
            return

    colourise_log(log_line, use_console=use_console)
    if log_line.lower() in error_messages:
        if use_console:
            console.render(log_line)
        else:
            logger.error(log_line)


def colourise_log(
    log: str,
    use_console: bool = False,
):
    # Adjust regex to capture the log message after the delimiter
    global logger
    if logger is None:
        logger = get_logger()

    # Use \s* to match any amount of whitespace
    if re.search(r"\|\s*INFO\s*\|", log):
        log = re.sub(r"\|\s*INFO\s*\|(.*)", "", log)
        if use_console:
            console.print(log.strip())
        else:
            logger.info(log.strip())
        return
    elif re.search(r"\|\|\s*DEBUG\s*\|\|", log):
        if os.getenv("ENV") != "prod":  # we only log debut on dev/local
            log = re.sub(r"\|\|\s*DEBUG\s*\|\|(.*)\|\|\s*END DEBUG\s*\|\|", r"\1", log)
            formatted = f"[yellow]{log.strip()}[/yellow]"
            if use_console:
                console.print(formatted)
            else:
                logger.info(formatted)
            return
    elif re.search(r"\|\|\s*ERROR\s*\|\|", log) or "ERROR:" in log:
        log = re.sub(r"\|\|\s*ERROR\s*\|\|(.*)\|\|\s*END ERROR\s*\|\|", r"\1", log)
        log = log.replace("ERROR:", "").strip()
        formatted = f"[red]{log.strip()}[/red]"
        if use_console:
            console.print(formatted)
        else:
            logger.error(formatted)
        return

    # logger.info(escaped_message)
    if use_console:
        console.print(log.strip())
    else:
        logger.info(log.strip())


def cerebrium_log(
    message: str,
    prefix: str = "",
    level: Union[
        Literal["DEBUG"], Literal["INFO"], Literal["WARNING"], Literal["ERROR"]
    ] = "INFO",
    attrs: Iterable[Attribute] = [],
    color: Color = "cyan",
    prefix_seperator: str = "\t",
    spinner: Union[Yaspin, None] = None,
    exit_on_error: bool = True,
):
    """User friendly coloured logging

    Args:
        message (str): Error message to be displayed
        prefix (str): Prefix to be displayed. Defaults to empty.
        level (str): Log level. Defaults to "INFO".
        attrs (list, optional): Attributes for colored printing. Defaults to None.
        color (str, optional): Color to print in. Defaults depending on log level.
        end (str, optional): End character. Defaults to "\n".
    """
    log_level = level.upper()
    default_prefixes = {
        "DEBUG": "Debug",
        "INFO": "Info: ",
        "WARNING": "Warning: ",
        "ERROR": "Error: ",
    }
    default_colors: Dict[str, Color] = {
        "DEBUG": "grey",
        "INFO": "cyan",
        "WARNING": "yellow",
        "ERROR": "red",
    }
    prefix = prefix or default_prefixes.get(log_level, "")

    # None is default for unused variables to avoid breaking termcolor
    log_color = color or default_colors.get(level, "cyan")
    prefix = colored(f"{prefix}", color=log_color, attrs=["bold"])
    message = colored(f"{message}", color=log_color, attrs=attrs)

    # spinners don't print nicely and keep spinning on errors. Use them if they're there
    if spinner:
        spinner.write(prefix)  # type: ignore
        spinner.text = ""
        if level == "ERROR":
            spinner.fail(message)
            spinner.stop()
        else:
            spinner.write(message)  # type: ignore
    else:
        print(prefix, end=prefix_seperator)
        print(message)

    if level == "ERROR" and exit_on_error:
        sys.exit(1)
