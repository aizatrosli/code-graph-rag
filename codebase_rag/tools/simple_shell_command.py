import asyncio
import shlex
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, cast

from langchain_core.tools import tool
from loguru import logger

from ..schemas import ShellCommandResult

# A strict list of commands the agent is allowed to execute.
COMMAND_ALLOWLIST = {
    "ls",
    "rg",
    "cat",
    "echo",
    "pwd",
    "pytest",
    "mypy",
    "ruff",
    "find",
}


def timing_decorator(
    func: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """
    A decorator that logs the execution time of the decorated asynchronous function.
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        logger.info(f"'{func.__qualname__}' executed in {execution_time:.2f}ms")
        return result

    return wrapper


class ShellCommander:
    """Service to execute shell commands."""

    def __init__(self, project_root: str = ".", timeout: int = 30):
        self.project_root = Path(project_root).resolve()
        self.timeout = timeout
        logger.info(f"ShellCommander initialized with root: {self.project_root}")

    @timing_decorator
    async def execute(
        self, command: str, confirmed: bool = False
    ) -> ShellCommandResult:
        """
        Execute a shell command and return the status code, stdout, and stderr.
        """
        logger.info(f"Executing shell command: {command}")
        try:
            cmd_parts = shlex.split(command)
            if not cmd_parts:
                return ShellCommandResult(
                    return_code=-1, stdout="", stderr="Empty command provided."
                )

            # Security: Check if the command is in the allowlist
            if cmd_parts[0] not in COMMAND_ALLOWLIST:
                available_commands = ", ".join(sorted(COMMAND_ALLOWLIST))
                suggestion = ""
                if cmd_parts[0] == "grep":
                    suggestion = " Use 'rg' instead of 'grep' for text searching."

                err_msg = f"Command '{cmd_parts[0]}' is not in the allowlist.{suggestion} Available commands: {available_commands}"
                logger.error(err_msg)
                return ShellCommandResult(return_code=-1, stdout="", stderr=err_msg)

            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout
            )

            stdout_str = stdout.decode("utf-8", errors="replace").strip()
            stderr_str = stderr.decode("utf-8", errors="replace").strip()

            logger.info(f"Return code: {process.returncode}")
            if stdout_str:
                logger.info(f"Stdout: {stdout_str}")
            if stderr_str:
                logger.warning(f"Stderr: {stderr_str}")

            return ShellCommandResult(
                return_code=(
                    process.returncode if process.returncode is not None else -1
                ),
                stdout=stdout_str,
                stderr=stderr_str,
            )
        except TimeoutError:
            msg = f"Command '{command}' timed out after {self.timeout} seconds."
            logger.error(msg)
            try:
                process.kill()
                await process.wait()
                logger.info("Process killed due to timeout.")
            except ProcessLookupError:
                logger.warning(
                    "Process already terminated when timeout kill was attempted."
                )
            return ShellCommandResult(return_code=-1, stdout="", stderr=msg)
        except Exception as e:
            logger.error(f"An error occurred while executing command: {e}")
            return ShellCommandResult(return_code=-1, stdout="", stderr=str(e))


def create_shell_command_tool(shell_commander: ShellCommander):
    """Factory function to create the shell command tool."""

    @tool
    async def run_shell_command(
        command: Annotated[str, "The shell command to execute"],
        user_confirmed: Annotated[bool, "Set to True if user has explicitly confirmed this command"] = False
    ) -> ShellCommandResult:
        """
        Executes a shell command from the approved allowlist only.

        Args:
            command: The shell command to execute
            user_confirmed: Set to True if user has explicitly confirmed this command

        AVAILABLE COMMANDS:
        - File operations: ls, cat, find, pwd
        - Text search: rg (ripgrep) - USE THIS INSTEAD OF grep
        - Testing: pytest, mypy, ruff
        - Other: echo

        IMPORTANT: Use 'rg' for text searching, NOT 'grep' (grep is not available).

        """
        return cast(
            ShellCommandResult,
            await shell_commander.execute(command, confirmed=user_confirmed),
        )

    return run_shell_command
