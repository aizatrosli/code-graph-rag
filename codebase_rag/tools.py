import asyncio
import difflib
import os
import shlex
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, TypedDict, cast

from loguru import logger
from pydantic import BaseModel
from tree_sitter import Node, Parser
from langchain_core.tools import tool
import diff_match_patch

from .graph_updater import MemgraphIngestor
from .language_config import get_language_config
from .parser_loader import load_parsers
from .schemas import CodeSnippet, GraphData, ShellCommandResult
from .services import CypherGenerator, LLMGenerationError

class CodeRetriever:
    """Service to retrieve code snippets using the graph and filesystem."""

    def __init__(self, project_root: str, ingestor: MemgraphIngestor):
        self.project_root = Path(project_root).resolve()
        self.ingestor = ingestor
        logger.info(f"CodeRetriever initialized with root: {self.project_root}")

    async def find_code_snippet(self, qualified_name: str) -> CodeSnippet:
        """Finds a code snippet by querying the graph for its location."""
        logger.info(f"[CodeRetriever] Searching for: {qualified_name}")

        query = """
            MATCH (n) WHERE n.qualified_name = $qn
            OPTIONAL MATCH (m:Module)-[*]-(n)
            RETURN n.name AS name, n.start_line AS start, n.end_line AS end, m.path AS path, n.docstring AS docstring
            LIMIT 1
        """
        params = {"qn": qualified_name}
        try:
            # Use the ingestor's public interface
            results = self.ingestor.fetch_all(query, params)

            if not results:
                return CodeSnippet(
                    qualified_name=qualified_name,
                    source_code="",
                    file_path="",
                    line_start=0,
                    line_end=0,
                    found=False,
                    error_message="Entity not found in graph.",
                )

            res = results[0]
            file_path_str = res.get("path")
            start_line = res.get("start")
            end_line = res.get("end")

            if not all([file_path_str, start_line, end_line]):
                return CodeSnippet(
                    qualified_name=qualified_name,
                    source_code="",
                    file_path=file_path_str or "",
                    line_start=0,
                    line_end=0,
                    found=False,
                    error_message="Graph entry is missing location data.",
                )

            full_path = self.project_root / file_path_str
            with full_path.open("r", encoding="utf-8") as f:
                all_lines = f.readlines()

            snippet_lines = all_lines[start_line - 1 : end_line]
            source_code = "".join(snippet_lines)

            return CodeSnippet(
                qualified_name=qualified_name,
                source_code=source_code,
                file_path=file_path_str,
                line_start=start_line,
                line_end=end_line,
                docstring=res.get("docstring"),
            )
        except Exception as e:
            logger.error(f"[CodeRetriever] Error: {e}", exc_info=True)
            return CodeSnippet(
                qualified_name=qualified_name,
                source_code="",
                file_path="",
                line_start=0,
                line_end=0,
                found=False,
                error_message=str(e),
            )


def create_code_retrieval_tool(code_retriever: CodeRetriever):
    """Factory function to create the code snippet retrieval tool."""

    @tool
    async def get_code_snippet(
        qualified_name: Annotated[str, "Full qualified name of the function, class, or method"]
    ) -> CodeSnippet:
        """Retrieves the source code for a given qualified name."""
        logger.info(f"[Tool:GetCode] Retrieving code for: {qualified_name}")
        return await code_retriever.find_code_snippet(qualified_name)

    return get_code_snippet





class GraphQueryError(Exception):
    """Custom exception for graph query failures."""

    pass


def create_query_tool(
    ingestor: MemgraphIngestor,
    cypher_gen: CypherGenerator,
):
    """
    Factory function that creates the knowledge graph query tool,
    injecting its dependencies.
    """

    @tool
    async def query_codebase_knowledge_graph(
        natural_language_query: Annotated[str, "Natural language query about the codebase"]
    ) -> GraphData:
        """
        Queries the codebase knowledge graph using natural language.

        Provide your question in plain English about the codebase structure,
        functions, classes, dependencies, or relationships. The tool will
        automatically translate your natural language question into the
        appropriate database query and return the results.

        Examples:
        - "Find all functions that call each other"
        - "What classes are in the user authentication module"
        - "Show me functions with the longest call chains"
        - "Which files contain functions related to database operations"
        """
        logger.info(f"[Tool:QueryGraph] Received NL query: '{natural_language_query}'")
        cypher_query = "N/A"
        try:
            cypher_query = await cypher_gen.generate(natural_language_query)

            results = ingestor.fetch_all(cypher_query)

            summary = f"Successfully retrieved {len(results)} item(s) from the graph."
            return GraphData(query_used=cypher_query, results=results, summary=summary)
        except LLMGenerationError as e:
            return GraphData(
                query_used="N/A",
                results=[],
                summary=f"I couldn't translate your request into a database query. Error: {e}",
            )
        except Exception as e:
            logger.error(
                f"[Tool:QueryGraph] Error during query execution: {e}", exc_info=True
            )
            return GraphData(
                query_used=cypher_query,
                results=[],
                summary=f"There was an error querying the database: {e}",
            )

    return query_codebase_knowledge_graph

class DirectoryLister:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()

    def list_directory_contents(self, directory_path: str) -> str:
        """
        Lists the contents of a specified directory.
        """
        target_path = self._get_safe_path(directory_path)
        logger.info(f"Listing contents of directory: {target_path}")

        try:
            if not target_path.is_dir():
                return f"Error: '{directory_path}' is not a valid directory."

            if contents := os.listdir(target_path):
                return "\n".join(contents)
            else:
                return f"The directory '{directory_path}' is empty."

        except Exception as e:
            logger.error(f"Error listing directory {directory_path}: {e}")
            return f"Error: Could not list contents of '{directory_path}'."

    def _get_safe_path(self, file_path: str) -> Path:
        """
        Resolves the file path relative to the root and ensures it's within
        the project directory.
        """
        # Accommodate both relative and absolute paths from the agent
        if Path(file_path).is_absolute():
            # If absolute, it should still be within the root path
            safe_path = Path(file_path).resolve()
        else:
            # If relative, resolve it against the root path
            safe_path = (self.project_root / file_path).resolve()

        try:
            safe_path.relative_to(self.project_root)
        except ValueError as e:
            raise PermissionError(
                "Access denied: Cannot access files outside the project root."
            ) from e

        return safe_path


def create_directory_lister_tool(directory_lister: DirectoryLister):
    @tool
    def list_directory_contents(
        directory_path: Annotated[str, "Path to the directory to list"]
    ) -> str:
        """Lists the contents of a directory to explore the codebase."""
        return directory_lister.list_directory_contents(directory_path)
    
    return list_directory_contents


class FunctionMatch(TypedDict):
    node: Node
    simple_name: str
    qualified_name: str
    parent_class: str | None
    line_number: int


LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".scala": "scala",
    ".cpp": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hxx": "cpp",
    ".hh": "cpp",
}


class EditResult(BaseModel):
    """Data model for file edit results."""

    file_path: str
    success: bool
    error_message: str | None = None


class FileEditor:
    def __init__(self, project_root: str = ".") -> None:
        self.project_root = Path(project_root).resolve()
        self.dmp = diff_match_patch.diff_match_patch()
        # Load parsers using the shared parser loader
        self.parsers, _ = load_parsers()
        logger.info(f"FileEditor initialized with root: {self.project_root}")

    def _get_real_extension(self, file_path_obj: Path) -> str:
        """Gets the file extension, looking past a .tmp suffix if present."""
        extension = file_path_obj.suffix
        if extension == ".tmp":
            # Get the extension before .tmp (e.g., test_file.py.tmp -> .py)
            base_name = file_path_obj.stem
            if "." in base_name:
                return "." + base_name.split(".")[-1]
        return extension

    def get_parser(self, file_path: str) -> Parser | None:
        file_path_obj = Path(file_path)
        extension = self._get_real_extension(file_path_obj)

        lang_name = LANGUAGE_EXTENSIONS.get(extension)
        if lang_name:
            return self.parsers.get(lang_name)
        return None

    def get_ast(self, file_path: str) -> Node | None:
        parser = self.get_parser(file_path)
        if not parser:
            logger.warning(f"No parser available for {file_path}")
            return None

        with open(file_path, "rb") as f:
            content = f.read()

        tree = parser.parse(content)
        return tree.root_node

    def get_function_source_code(
        self, file_path: str, function_name: str, line_number: int | None = None
    ) -> str | None:
        root_node = self.get_ast(file_path)
        if not root_node:
            return None

        # Get language config for this file
        file_path_obj = Path(file_path)
        extension = self._get_real_extension(file_path_obj)

        lang_config = get_language_config(extension)
        if not lang_config:
            logger.warning(f"No language config found for extension {extension}")
            return None

        # Find all matching functions with their context
        matching_functions: list[FunctionMatch] = []

        def find_function_nodes(node: Node, parent_class: str | None = None) -> None:
            if node.type in lang_config.function_node_types:
                # Get the function name node using the 'name' field
                name_node = node.child_by_field_name("name")
                if name_node and name_node.text:
                    func_name = name_node.text.decode("utf-8")

                    # Check if this matches our target function
                    qualified_name = (
                        f"{parent_class}.{func_name}" if parent_class else func_name
                    )

                    # Match either simple name or qualified name
                    if func_name == function_name or qualified_name == function_name:
                        matching_functions.append(
                            {
                                "node": node,
                                "simple_name": func_name,
                                "qualified_name": qualified_name,
                                "parent_class": parent_class,
                                "line_number": node.start_point[0]
                                + 1,  # 1-based line numbers
                            }
                        )

                    # Don't recurse into function bodies for nested functions
                    return

            # Check if this is a class node to track context
            current_class = parent_class
            if node.type in lang_config.class_node_types:
                name_node = node.child_by_field_name("name")
                if name_node and name_node.text:
                    current_class = name_node.text.decode("utf-8")

            # Recursively search children
            for child in node.children:
                find_function_nodes(child, current_class)

        find_function_nodes(root_node)

        # Handle different matching scenarios
        if not matching_functions:
            return None
        elif len(matching_functions) == 1:
            node_text = matching_functions[0]["node"].text
            if node_text is None:
                return None
            return str(node_text.decode("utf-8"))
        else:
            # Multiple functions found - try different disambiguation strategies

            # Strategy 1: Match by line number if provided
            if line_number is not None:
                for func in matching_functions:
                    if func["line_number"] == line_number:
                        node_text = func["node"].text
                        if node_text is None:
                            return None
                        return str(node_text.decode("utf-8"))
                logger.warning(
                    f"No function '{function_name}' found at line {line_number}"
                )
                return None

            # Strategy 2: Match by qualified name if function_name contains dot
            if "." in function_name:
                for func in matching_functions:
                    if func["qualified_name"] == function_name:
                        node_text = func["node"].text
                        if node_text is None:
                            return None
                        return str(node_text.decode("utf-8"))
                logger.warning(
                    f"No function found with qualified name '{function_name}'"
                )
                return None

            # Strategy 3: Log ambiguity warning with details and return first match
            function_details = []
            for func in matching_functions:
                details = f"'{func['qualified_name']}' at line {func['line_number']}"
                function_details.append(details)

            logger.warning(
                f"Ambiguous function name '{function_name}' in {file_path}. "
                f"Found {len(matching_functions)} matches: {', '.join(function_details)}. "
                f"Using first match. Consider using qualified name (e.g., 'ClassName.{function_name}') "
                f"or specify line number for precise targeting."
            )

            # Return the first match but warn the user
            node_text = matching_functions[0]["node"].text
            if node_text is None:
                return None
            return str(node_text.decode("utf-8"))

    def replace_function_source_code(
        self,
        file_path: str,
        function_name: str,
        new_code: str,
        line_number: int | None = None,
    ) -> bool:
        original_code = self.get_function_source_code(
            file_path, function_name, line_number
        )
        if not original_code:
            logger.error(f"Function '{function_name}' not found in {file_path}.")
            return False

        with open(file_path, encoding="utf-8") as f:
            original_content = f.read()

        # Create patches using diff-match-patch
        patches = self.dmp.patch_make(original_code, new_code)

        # Apply patches to the original content
        new_content, results = self.dmp.patch_apply(patches, original_content)

        # Check if all patches were applied successfully
        if not all(results):
            logger.warning(
                f"Patches for function '{function_name}' did not apply cleanly."
            )
            return False

        if original_content == new_content:
            logger.warning("No changes detected after replacement.")
            return False

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        logger.success(
            f"Successfully replaced function '{function_name}' in {file_path}."
        )
        return True

    def get_diff(
        self,
        file_path: str,
        function_name: str,
        new_code: str,
        line_number: int | None = None,
    ) -> str | None:
        original_code = self.get_function_source_code(
            file_path, function_name, line_number
        )
        if not original_code:
            return None

        # Use diff-match-patch for more sophisticated diff generation
        diffs = self.dmp.diff_main(original_code, new_code)
        self.dmp.diff_cleanupSemantic(diffs)

        # Convert to unified diff format for readability
        diff = difflib.unified_diff(
            original_code.splitlines(keepends=True),
            new_code.splitlines(keepends=True),
            fromfile=f"original/{function_name}",
            tofile=f"new/{function_name}",
        )
        return "".join(diff)

    def apply_patch_to_file(self, file_path: str, patch_text: str) -> bool:
        """Apply a patch to a file using diff-match-patch."""
        try:
            with open(file_path, encoding="utf-8") as f:
                original_content = f.read()

            # Parse the patch
            patches = self.dmp.patch_fromText(patch_text)

            # Apply the patch
            new_content, results = self.dmp.patch_apply(patches, original_content)

            # Check if all patches were applied successfully
            if not all(results):
                logger.warning(f"Some patches failed to apply cleanly to {file_path}")
                return False

            # Write the updated content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            logger.success(f"Successfully applied patch to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error applying patch to {file_path}: {e}")
            return False

    def _display_colored_diff(
        self, original_content: str, new_content: str, file_path: str
    ) -> None:
        """Display a concise colored diff with limited context."""
        # Generate diffs using diff-match-patch
        diffs = self.dmp.diff_main(original_content, new_content)
        self.dmp.diff_cleanupSemantic(diffs)

        # ANSI color codes
        RED = "\033[31m"
        GREEN = "\033[32m"
        CYAN = "\033[36m"
        GRAY = "\033[90m"
        RESET = "\033[0m"

        print(f"\n{CYAN}Changes to {file_path}:{RESET}")

        CONTEXT_LINES = 5  # Show 5 lines before/after changes

        # Process diffs to show limited context
        for op, text in diffs:
            # Use keepends=True to preserve newlines for accurate rendering
            lines = text.splitlines(keepends=True)

            if op == self.dmp.DIFF_DELETE:
                for line in lines:
                    # rstrip to remove the trailing newline for cleaner printing
                    print(f"{RED}- {line.rstrip()}{RESET}")
            elif op == self.dmp.DIFF_INSERT:
                for line in lines:
                    print(f"{GREEN}+ {line.rstrip()}{RESET}")
            elif op == self.dmp.DIFF_EQUAL:
                # For unchanged sections, show limited context
                if len(lines) > CONTEXT_LINES * 2:
                    # Show first few lines
                    for line in lines[:CONTEXT_LINES]:
                        print(f"  {line.rstrip()}")

                    # Show truncation indicator if there are many lines
                    omitted_count = len(lines) - (CONTEXT_LINES * 2)
                    if omitted_count > 0:
                        print(f"{GRAY}  ... ({omitted_count} lines omitted) ...{RESET}")

                    # Show last few lines
                    for line in lines[-CONTEXT_LINES:]:
                        print(f"  {line.rstrip()}")
                else:
                    # Show all lines if not too many
                    for line in lines:
                        print(f"  {line.rstrip()}")

        print()  # Extra newline for spacing

    def replace_code_block(
        self, file_path: str, target_block: str, replacement_block: str
    ) -> bool:
        """Surgically replace a specific code block in a file using diff-match-patch."""
        logger.info(
            f"[FileEditor] Attempting surgical block replacement in: {file_path}"
        )
        try:
            full_path = (self.project_root / file_path).resolve()
            full_path.relative_to(self.project_root)  # Security check

            if not full_path.is_file():
                logger.error(f"File not found: {file_path}")
                return False

            # Read original content
            with open(full_path, encoding="utf-8") as f:
                original_content = f.read()

            # Find the target block in the file
            if target_block not in original_content:
                logger.error(f"Target block not found in {file_path}")
                logger.debug(f"Looking for: {repr(target_block)}")
                return False

            # Create surgical patch - replace only the target block
            modified_content = original_content.replace(
                target_block, replacement_block, 1
            )

            # Verify only one replacement was made
            if original_content.count(target_block) > 1:
                logger.warning(
                    "Multiple occurrences of target block found. Only replacing first occurrence."
                )

            if original_content == modified_content:
                logger.warning(
                    "No changes detected - target and replacement are identical"
                )
                return False

            # Display the surgical diff
            self._display_colored_diff(original_content, modified_content, file_path)

            # Create and apply surgical patches
            patches = self.dmp.patch_make(original_content, modified_content)
            patched_content, results = self.dmp.patch_apply(patches, original_content)

            if not all(results):
                logger.error("Surgical patches failed to apply cleanly")
                return False

            # Write the surgically modified content
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(patched_content)

            logger.success(
                f"[FileEditor] Successfully applied surgical block replacement in: {file_path}"
            )
            return True

        except ValueError:
            logger.error(
                "Security risk: Attempted to edit file outside of project root."
            )
            return False
        except Exception as e:
            logger.error(f"Error during surgical block replacement: {e}")
            return False

    async def edit_file(self, file_path: str, new_content: str) -> EditResult:
        """Overwrites entire file with new content - use for full file replacement."""
        logger.info(f"[FileEditor] Attempting full file replacement: {file_path}")
        try:
            full_path = (self.project_root / file_path).resolve()
            full_path.relative_to(self.project_root)  # Security check

            # Check if the file exists and is a file before proceeding
            if not full_path.is_file():
                error_msg = f"File not found or is a directory: {file_path}"
                logger.warning(f"[FileEditor] {error_msg}")
                return EditResult(
                    file_path=file_path, success=False, error_message=error_msg
                )

            # Read original content to show diff
            with open(full_path, encoding="utf-8") as f:
                original_content = f.read()

            # Display colored diff
            if original_content != new_content:
                self._display_colored_diff(original_content, new_content, file_path)

            # Write new content (full replacement)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            logger.success(
                f"[FileEditor] Successfully replaced entire file: {file_path}"
            )
            return EditResult(file_path=file_path, success=True)

        except ValueError:
            error_msg = "Security risk: Attempted to edit file outside of project root."
            logger.error(f"[FileEditor] {error_msg}")
            return EditResult(
                file_path=file_path, success=False, error_message=error_msg
            )
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            logger.error(f"[FileEditor] Error editing file {file_path}: {e}")
            return EditResult(
                file_path=file_path, success=False, error_message=error_msg
            )


def create_file_editor_tool(file_editor: FileEditor):
    """Factory function to create the file editor tool."""

    @tool
    async def replace_code_surgically(
        file_path: Annotated[str, "Path to the file to modify"],
        target_code: Annotated[str, "The exact code block to find and replace (must match exactly)"],
        replacement_code: Annotated[str, "The new code to replace the target with"]
    ) -> str:
        """
        Surgically replaces a specific code block in a file using diff-match-patch.
        This tool finds the exact target code block and replaces only that section,
        leaving the rest of the file completely unchanged. This is true surgical patching.

        Args:
            file_path: Path to the file to modify
            target_code: The exact code block to find and replace (must match exactly)
            replacement_code: The new code to replace the target with

        Use this when you need to change specific functions, classes, or code blocks
        without affecting the rest of the file. The target_code must be an exact match.
        """
        success = file_editor.replace_code_block(
            file_path, target_code, replacement_code
        )
        if success:
            return f"Successfully applied surgical code replacement in: {file_path}"
        else:
            return f"Failed to apply surgical replacement in {file_path}. Target code not found or patches failed."

    return replace_code_surgically


class FileReadResult(BaseModel):
    """Data model for file read results."""

    file_path: str
    content: str | None = None
    error_message: str | None = None


class FileReader:
    """Service to read file content from the filesystem."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        # Define extensions that should be treated as binary and not read by this tool
        self.binary_extensions = {
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".ico",
            ".tiff",
            ".webp",
        }
        logger.info(f"FileReader initialized with root: {self.project_root}")

    async def read_file(self, file_path: str) -> FileReadResult:
        """Reads and returns the content of a text-based file."""
        logger.info(f"[FileReader] Attempting to read file: {file_path}")
        try:
            full_path = (self.project_root / file_path).resolve()
            full_path.relative_to(self.project_root)  # Security check

            if not full_path.is_file():
                return FileReadResult(
                    file_path=file_path, error_message="File not found."
                )

            # Check if the file has a binary extension
            if full_path.suffix.lower() in self.binary_extensions:
                error_msg = f"File '{file_path}' is a binary file. Use the 'analyze_document' tool for this file type."
                logger.warning(f"[FileReader] {error_msg}")
                return FileReadResult(file_path=file_path, error_message=error_msg)

            # Proceed with reading as a text file
            try:
                content = full_path.read_text(encoding="utf-8")
                logger.info(f"[FileReader] Successfully read text from {file_path}")
                return FileReadResult(file_path=file_path, content=content)
            except UnicodeDecodeError:
                error_msg = f"File '{file_path}' could not be read as text. It may be a binary file. If it is a document (e.g., PDF), use the 'analyze_document' tool."
                logger.warning(f"[FileReader] {error_msg}")
                return FileReadResult(file_path=file_path, error_message=error_msg)

        except ValueError:
            return FileReadResult(
                file_path=file_path,
                error_message="Security risk: Attempted to read file outside of project root.",
            )
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return FileReadResult(
                file_path=file_path, error_message=f"An unexpected error occurred: {e}"
            )


def create_file_reader_tool(file_reader: FileReader):
    """Factory function to create the file reader tool."""

    @tool
    async def read_file_content(
        file_path: Annotated[str, "Path to the text-based file to read"]
    ) -> str:
        """
        Reads the content of a specified text-based file (e.g., source code, README.md, config files).
        This tool should NOT be used for binary files like PDFs or images. For those, use the 'analyze_document' tool.
        """
        result = await file_reader.read_file(file_path)
        if result.error_message:
            return f"Error: {result.error_message}"
        return result.content or ""

    return read_file_content


class FileCreationResult(BaseModel):
    """Data model for file creation results."""

    file_path: str
    success: bool = True
    error_message: str | None = None


class FileWriter:
    """Service to write file content to the filesystem."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        logger.info(f"FileWriter initialized with root: {self.project_root}")

    async def create_file(self, file_path: str, content: str) -> FileCreationResult:
        """Creates or overwrites a file with the given content."""
        logger.info(f"[FileWriter] Creating file: {file_path}")
        try:
            # Resolve the path to prevent traversal attacks
            full_path = (self.project_root / file_path).resolve()

            # Security check: Ensure the resolved path is within the project root
            full_path.relative_to(self.project_root)

            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            logger.info(
                f"[FileWriter] Successfully wrote {len(content)} characters to {file_path}"
            )
            return FileCreationResult(file_path=file_path)
        except ValueError:
            err_msg = f"Security risk: Attempted to create file outside of project root: {file_path}"
            logger.error(err_msg)
            return FileCreationResult(
                file_path=file_path, success=False, error_message=err_msg
            )
        except Exception as e:
            err_msg = f"Error creating file {file_path}: {e}"
            logger.error(err_msg)
            return FileCreationResult(
                file_path=file_path, success=False, error_message=err_msg
            )


def create_file_writer_tool(file_writer: FileWriter):
    """Factory function to create the file writer tool."""

    @tool
    async def create_new_file(
        file_path: Annotated[str, "Path where the new file should be created"],
        content: Annotated[str, "Content to write to the new file"]
    ) -> FileCreationResult:
        """
        Creates a new file with the specified content.

        IMPORTANT: Before using this tool, you MUST check if the file already exists using
        the file reader or directory listing tools. If the file exists, use edit_existing_file
        instead to preserve existing content and show diffs.

        If the file already exists, it will be completely overwritten WITHOUT showing any diff.
        Use this ONLY for creating entirely new files, not for modifying existing ones.
        For modifying existing files with diff preview, use edit_existing_file instead.
        """
        return await file_writer.create_file(file_path, content)

    return create_new_file


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


