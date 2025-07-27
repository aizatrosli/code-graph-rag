import os

from typing import Literal
from dataclasses import dataclass, field, replace, asdict
from dotenv import load_dotenv
from prompt_toolkit.styles import Style
from langchain_core.language_models.llms import BaseLLM
from langchain_openai import AzureChatOpenAI, ChatOpenAI

load_dotenv()

@dataclass
class CodebaseConfig:
    MEMGRAPH_HOST: str | None = None
    MEMGRAPH_PORT: int | None = None
    MEMGRAPH_HTTP_PORT: int | None = None
    LAB_PORT: int | None = None

    # Azure OpenAI Configuration
    CODEBASE_PROVIDER: str | None = None
    CODEBASE_API_KEY: str | None = None
    CODEBASE_ENDPOINT: str | None = None
    CODEBASE_API_VERSION: str | None = None
    CODEBASE_ORCHESTRATOR_DEPLOYMENT: str | None = None
    CODEBASE_CYPHER_DEPLOYMENT: str | None = None

    TARGET_REPO_PATH: str | None = None
    SHELL_COMMAND_TIMEOUT: int | None = None
    RECURSION_LIMIT: int | None = None

    LANGFUSE_HOST: str = field(default_factory=lambda: os.environ.get("LANGFUSE_HOST", "http://"))
    LANGFUSE_PUBLIC_KEY: str = field(default_factory=lambda: os.environ.get("LANGFUSE_PUBLIC_KEY", ""))
    LANGFUSE_SECRET_KEY: str = field(default_factory=lambda: os.environ.get("LANGFUSE_SECRET_KEY", ""))
    LANGFUSE_PROJECT: str = field(default_factory=lambda: os.environ.get("LANGFUSE_PROJECT", "CodeRAG"))

    ACTIVE_ORCHESTRATOR_MODEL: BaseLLM | None = None
    ACTIVE_CYPHER_MODEL: BaseLLM | None = None
    def __post_init__(self):
        self.MEMGRAPH_HOST = os.environ.get("MEMGRAPH_HOST", "localhost") if self.MEMGRAPH_HOST is None else self.MEMGRAPH_HOST
        self.MEMGRAPH_PORT = int(os.environ.get("MEMGRAPH_PORT", 7687)) if self.MEMGRAPH_PORT is None else self.MEMGRAPH_PORT
        self.MEMGRAPH_HTTP_PORT = int(os.environ.get("MEMGRAPH_HTTP_PORT", 7444)) if self.MEMGRAPH_HTTP_PORT is None else self.MEMGRAPH_HTTP_PORT
        self.LAB_PORT = int(os.environ.get("LAB_PORT", 3000)) if self.LAB_PORT is None else self.LAB_PORT
        self.CODEBASE_PROVIDER = os.environ.get("CODEBASE_PROVIDER", "azure") if self.CODEBASE_PROVIDER is None else self.CODEBASE_PROVIDER
        self.CODEBASE_API_KEY = os.environ.get("CODEBASE_API_KEY") if self.CODEBASE_API_KEY is None else self.CODEBASE_API_KEY
        self.CODEBASE_ENDPOINT = os.environ.get("CODEBASE_ENDPOINT") if self.CODEBASE_ENDPOINT is None else self.CODEBASE_ENDPOINT
        self.CODEBASE_API_VERSION = os.environ.get("CODEBASE_API_VERSION", "2024-02-01") if self.CODEBASE_API_VERSION is None else self.CODEBASE_API_VERSION
        self.CODEBASE_ORCHESTRATOR_DEPLOYMENT = os.environ.get("CODEBASE_ORCHESTRATOR_DEPLOYMENT", "gpt-4o-mini") if self.CODEBASE_ORCHESTRATOR_DEPLOYMENT is None else self.CODEBASE_ORCHESTRATOR_DEPLOYMENT
        self.CODEBASE_CYPHER_DEPLOYMENT = os.environ.get("CODEBASE_CYPHER_DEPLOYMENT", "gpt-35-turbo") if self.CODEBASE_CYPHER_DEPLOYMENT is None else self.CODEBASE_CYPHER_DEPLOYMENT
        self.TARGET_REPO_PATH = os.environ.get("TARGET_REPO_PATH", ".") if self.TARGET_REPO_PATH is None else self.TARGET_REPO_PATH
        self.SHELL_COMMAND_TIMEOUT = int(os.environ.get("SHELL_COMMAND_TIMEOUT", "30")) if self.SHELL_COMMAND_TIMEOUT is None else self.SHELL_COMMAND_TIMEOUT
        self.RECURSION_LIMIT = int(os.environ.get("RECURSION_LIMIT", "20")) if self.RECURSION_LIMIT is None else self.RECURSION_LIMIT
    
        if self.CODEBASE_PROVIDER == "azure":
            # Extract deployment name from model ID (azure-deployment-name format)
            headers = {'Ocp-Apim-Subscription-Key': self.CODEBASE_API_KEY}
            self.ACTIVE_ORCHESTRATOR_MODEL = AzureChatOpenAI(
                azure_endpoint=self.CODEBASE_ENDPOINT,
                api_key='dummy',
                azure_deployment=self.CODEBASE_ORCHESTRATOR_DEPLOYMENT,
                api_version=self.CODEBASE_API_VERSION,
                default_headers=headers
            )
            self.ACTIVE_CYPHER_MODEL = AzureChatOpenAI(
                azure_endpoint=self.CODEBASE_ENDPOINT,
                api_key='dummy',
                azure_deployment=self.CODEBASE_CYPHER_DEPLOYMENT,
                api_version=self.CODEBASE_API_VERSION,
                default_headers=headers
            )
        else:
            self.ACTIVE_ORCHESTRATOR_MODEL = ChatOpenAI(
                model=self.CODEBASE_ORCHESTRATOR_DEPLOYMENT,
                api_key=self.CODEBASE_API_KEY,
                base_url=self.CODEBASE_ENDPOINT,
                temperature=0.0,
            )
            self.ACTIVE_ORCHESTRATOR_MODEL = ChatOpenAI(
                model=self.CODEBASE_ORCHESTRATOR_DEPLOYMENT,
                api_key=self.CODEBASE_API_KEY,
                base_url=self.CODEBASE_ENDPOINT,
                temperature=0.0,
            )


# --- Global Ignore Patterns ---
# Directories and files to ignore during codebase scanning and real-time updates.
IGNORE_PATTERNS = {
    ".git",
    "venv",
    ".venv",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
    ".eggs",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".claude",
    ".idea",
    ".vscode",
}
IGNORE_SUFFIXES = {".tmp", "~"}


# --- Edit Operation Constants ---
# Keywords that might indicate a user wants to perform an edit operation.
EDIT_REQUEST_KEYWORDS = frozenset(
    [
        "modify",
        "update",
        "change",
        "edit",
        "fix",
        "refactor",
        "optimize",
        "add",
        "remove",
        "delete",
        "create",
        "write",
        "implement",
        "replace",
    ]
)

# Tool names that are considered edit operations.
EDIT_TOOLS = frozenset(
    [
        "edit_file",
        "write_file",
        "file_editor",
        "file_writer",
        "create_file",
        "replace_code_surgically",
    ]
)

# Phrases in a model's response that indicate an edit has been performed.
EDIT_INDICATORS = frozenset(
    [
        "modifying",
        "updating",
        "changing",
        "replacing",
        "adding to",
        "deleting from",
        "created file",
        "editing",
        "writing to",
        "file has been",
        "successfully modified",
        "successfully updated",
        "successfully created",
        "changes have been made",
        "file modified",
        "file updated",
        "file created",
    ]
)

# --- UI Styles ---
# Style for user input prompts in the terminal.
ORANGE_STYLE = Style.from_dict({"": "#ff8c00"})
