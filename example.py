"""
Simplified Code Graph RAG - Non-interactive programmatic interface

This module provides a simplified, non-interactive interface for the Code Graph RAG system.
It removes all CLI dependencies, logging, and console interactions for programmatic usage.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from codebase_rag.config import CodebaseConfig
from codebase_rag.graph_updater import GraphUpdater, MemgraphIngestor
from codebase_rag.parser_loader import load_parsers
from codebase_rag.rag_orchestrator import RAGOrchestrator
from codebase_rag.services.llm_langgraph import CypherGenerator
from codebase_rag.tools.code_retrieval import CodeRetriever, create_code_retrieval_tool
from codebase_rag.tools.codebase_query import create_query_tool
from codebase_rag.tools.directory_lister import DirectoryLister, create_directory_lister_tool
from codebase_rag.tools.file_editor import FileEditor, create_file_editor_tool
from codebase_rag.tools.file_reader import FileReader, create_file_reader_tool
from codebase_rag.tools.file_writer import FileWriter, create_file_writer_tool
from codebase_rag.tools.simple_shell_command import ShellCommander, create_shell_command_tool


class CodeGraphRAG:
    """
    Simplified Code Graph RAG interface for programmatic usage.
    
    This class provides a clean, non-interactive interface to the RAG system
    without any CLI dependencies or console interactions.
    """
    
    def __init__(
        self,
        settings: CodebaseConfig,
    ):
        """
        Initialize the Code Graph RAG system.
        
        Args:
            repo_path: Path to the target repository
            memgraph_host: Memgraph database host
            memgraph_port: Memgraph database port
            orchestrator_model: Optional orchestrator model override
            cypher_model: Optional cypher model override
        """
        self.settings = settings
        self.repo_path = Path(settings.TARGET_REPO_PATH).resolve()
        self.memgraph_host = settings.MEMGRAPH_HOST
        self.memgraph_port = settings.MEMGRAPH_PORT
        self.ingestor = None
        self.rag_agent = None
    
    def __enter__(self):
        """Context manager entry - initialize database connection."""
        self.ingestor = MemgraphIngestor(
            host=self.memgraph_host, 
            port=self.memgraph_port
        )
        self.ingestor.__enter__()
        self._initialize_rag_agent()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup database connection."""
        if self.ingestor:
            self.ingestor.__exit__(exc_type, exc_val, exc_tb)
    
    def _initialize_rag_agent(self) -> None:
        """Initialize the RAG agent with all tools."""
        if not self.ingestor:
            raise RuntimeError("Database connection not established. Use as context manager.")
        
        # Initialize services
        cypher_generator = CypherGenerator(llm=self.settings.ACTIVE_CYPHER_MODEL)
        code_retriever = CodeRetriever(project_root=str(self.repo_path), ingestor=self.ingestor)
        file_reader = FileReader(project_root=str(self.repo_path))
        file_writer = FileWriter(project_root=str(self.repo_path))
        file_editor = FileEditor(project_root=str(self.repo_path))
        shell_commander = ShellCommander(project_root=str(self.repo_path), timeout=self.settings.SHELL_COMMAND_TIMEOUT)
        directory_lister = DirectoryLister(project_root=str(self.repo_path))
        
        # Create tools
        tools = [
            create_query_tool(self.ingestor, cypher_generator),
            create_code_retrieval_tool(code_retriever),
            create_file_reader_tool(file_reader),
            create_file_writer_tool(file_writer),
            create_file_editor_tool(file_editor),
            create_shell_command_tool(shell_commander),
            create_directory_lister_tool(directory_lister),
        ]
        
        # Initialize RAG orchestrator
        self.rag_agent = RAGOrchestrator(tools=tools, settings=self.settings)
    
    def update_knowledge_graph(self, clean_database: bool = False) -> bool:
        """
        Update the knowledge graph by parsing the repository.
        
        Args:
            clean_database: Whether to clean the database before updating
            
        Returns:
            True if successful, False otherwise
        """
        if not self.ingestor:
            raise RuntimeError("Database connection not established. Use as context manager.")
        
        try:
            if clean_database:
                self.ingestor.clean_database()
            self.ingestor.ensure_constraints()
            parsers, queries = load_parsers()
            
            # Update the graph
            updater = GraphUpdater(self.ingestor, self.repo_path, parsers, queries)
            updater.run()
            
            return True
        except Exception:
            return False
    
    def export_graph(self, output_path: str) -> bool:
        """
        Export the knowledge graph to a JSON file.
        
        Args:
            output_path: Path where to save the exported graph
            
        Returns:
            True if successful, False otherwise
        """
        if not self.ingestor:
            raise RuntimeError("Database connection not established. Use as context manager.")
        
        try:
            graph_data = self.ingestor.export_graph_to_dict()
            output_file = Path(output_path)
            
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write JSON
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception:
            return False
    
    async def query(self, question: str, message_history: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Query the codebase using natural language.
        
        Args:
            question: The natural language question
            message_history: Optional message history for context
            
        Returns:
            Dictionary containing the response and metadata
        """
        if not self.rag_agent:
            raise RuntimeError("RAG agent not initialized. Use as context manager.")
        
        if message_history is None:
            message_history = []
        
        try:
            response = await self.rag_agent.run(question, message_history)
            
            return {
                "success": True,
                "output": response.output,
                "new_messages": response.new_messages(),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "output": None,
                "new_messages": [],
                "error": str(e)
            }
    
    def query_sync(self, question: str, message_history: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Synchronous wrapper for the query method.
        
        Args:
            question: The natural language question
            message_history: Optional message history for context
            
        Returns:
            Dictionary containing the response and metadata
        """
        return asyncio.run(self.query(question, message_history))


class CodeOptimizer:
    """
    Specialized class for code optimization workflows.
    
    This class provides functionality similar to the optimize command
    but in a programmatic, non-interactive format.
    """
    
    def __init__(
        self,
        settings: CodebaseConfig,
        language: str,
        reference_document: Optional[str] = None,
    ):
        """
        Initialize the code optimizer.
        
        Args:
            repo_path: Path to the repository to optimize
            language: Programming language to optimize for
            reference_document: Optional path to reference document
            memgraph_host: Memgraph database host
            memgraph_port: Memgraph database port
            orchestrator_model: Optional orchestrator model override
            cypher_model: Optional cypher model override
        """
        self.rag_system = CodeGraphRAG(settings=settings)
        self.language = language
        self.reference_document = reference_document
    
    def __enter__(self):
        """Context manager entry."""
        self.rag_system.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.rag_system.__exit__(exc_type, exc_val, exc_tb)
    
    def _create_optimization_prompt(self) -> str:
        """Create the initial optimization analysis prompt."""
        instructions = [
            "Use your code retrieval and graph querying tools to understand the codebase structure",
            "Read relevant source files to identify optimization opportunities",
        ]
        
        if self.reference_document:
            instructions.append(
                f"Use the analyze_document tool to reference best practices from {self.reference_document}"
            )
        
        instructions.extend([
            f"Reference established patterns and best practices for {self.language}",
            "Propose specific, actionable optimizations with file references",
            "Use your file editing tools to implement the suggested changes",
        ])
        
        numbered_instructions = "\n".join(
            f"{i + 1}. {inst}" for i, inst in enumerate(instructions)
        )
        
        return f"""
I want you to analyze my {self.language} codebase and propose specific optimizations based on best practices.

Please:
{numbered_instructions}

Start by analyzing the codebase structure and identifying the main areas that could benefit from optimization.
"""
    
    async def analyze_and_optimize(self) -> Dict[str, Any]:
        """
        Perform automated code analysis and optimization.
        
        Returns:
            Dictionary containing optimization results
        """
        optimization_prompt = self._create_optimization_prompt()
        result = await self.rag_system.query(optimization_prompt)
        
        return {
            "language": self.language,
            "reference_document": self.reference_document,
            "analysis_result": result,
            "repo_path": str(self.rag_system.repo_path)
        }
    
    def analyze_and_optimize_sync(self) -> Dict[str, Any]:
        """
        Synchronous wrapper for analyze_and_optimize.
        
        Returns:
            Dictionary containing optimization results
        """
        return asyncio.run(self.analyze_and_optimize())


# Convenience functions for common operations

def create_knowledge_graph(
    repo_path: str,
    clean_database: bool = False,
    export_path: Optional[str] = None,
    memgraph_host: str = "localhost",
    memgraph_port: int = 7687
) -> bool:
    """
    Create or update a knowledge graph for a repository.
    
    Args:
        repo_path: Path to the repository
        clean_database: Whether to clean the database first
        export_path: Optional path to export the graph
        memgraph_host: Memgraph database host
        memgraph_port: Memgraph database port
        
    Returns:
        True if successful, False otherwise
    """
    settings = CodebaseConfig(
        MEMGRAPH_HOST=memgraph_host,
        MEMGRAPH_PORT=memgraph_port,
        TARGET_REPO_PATH=repo_path,
    )
    try:
        with CodeGraphRAG(settings=settings) as rag:
            success = rag.update_knowledge_graph(clean_database)
            
            if success and export_path:
                rag.export_graph(export_path)
            
            return success
    except Exception:
        return False


def query_codebase(
    repo_path: str,
    question: str,
    memgraph_host: str = "localhost",
    memgraph_port: int = 7687
) -> Dict[str, Any]:
    """
    Query a codebase with a natural language question.
    
    Args:
        repo_path: Path to the repository
        question: Natural language question
        memgraph_host: Memgraph database host
        memgraph_port: Memgraph database port
        
    Returns:
        Dictionary containing the response
    """
    settings = CodebaseConfig(
        MEMGRAPH_HOST=memgraph_host,
        MEMGRAPH_PORT=memgraph_port,
        TARGET_REPO_PATH=repo_path,
    )
    try:
        with CodeGraphRAG(settings=settings) as rag:
            return rag.query_sync(question)
    except Exception as e:
        return {"success": False, "output": None, "new_messages": [], "error": str(e)}


def optimize_codebase(
    repo_path: str,
    language: str,
    reference_document: Optional[str] = None,
    memgraph_host: str = "localhost",
    memgraph_port: int = 7687
) -> Dict[str, Any]:
    """
    Optimize a codebase for a specific programming language.
    
    Args:
        repo_path: Path to the repository
        language: Programming language to optimize for
        reference_document: Optional reference document path
        memgraph_host: Memgraph database host
        memgraph_port: Memgraph database port
        
    Returns:
        Dictionary containing optimization results
    """
    settings = CodebaseConfig(
        MEMGRAPH_HOST=memgraph_host,
        MEMGRAPH_PORT=memgraph_port,
        TARGET_REPO_PATH=repo_path,
    )
    try:
        with CodeOptimizer(settings=settings, language=language, reference_document=reference_document) as optimizer:
            return optimizer.analyze_and_optimize_sync()
    except Exception as e:
        return {"success": False, "error": str(e), "language": language, "repo_path": repo_path}



# Example usage functions

def example_basic_usage():
    """Example of basic usage patterns."""
    repo_path = "/path/to/your/repository"
    
    # Create knowledge graph
    success = create_knowledge_graph(
        repo_path=repo_path,
        clean_database=True,
        export_path="graph_export.json"
    )
    
    if success:
        print("Knowledge graph created successfully")
        
        # Query the codebase
        result = query_codebase(
            repo_path=repo_path,
            question="What are the main classes in this codebase?"
        )
        
        if result["success"]:
            print("Query result:", result["output"])
        else:
            print("Query failed:", result["error"])
    else:
        print("Failed to create knowledge graph")


def example_optimization_usage():
    """Example of optimization usage patterns."""
    repo_path = "/path/to/your/python/project"
    
    # Optimize Python codebase
    result = optimize_codebase(
        repo_path=repo_path,
        language="python",
        reference_document="/path/to/python_best_practices.pdf"
    )
    
    if result.get("analysis_result", {}).get("success"):
        print("Optimization analysis completed")
        print("Analysis:", result["analysis_result"]["output"])
    else:
        print("Optimization failed:", result.get("error"))


def example_advanced_usage():
    """Example of advanced usage with context management."""
    repo_path = "/path/to/your/repository"
    settings = CodebaseConfig(
        TARGET_REPO_PATH=repo_path,
    )
    # Advanced usage with multiple queries
    with CodeGraphRAG(settings=settings) as rag:
        # Update knowledge graph
        rag.update_knowledge_graph(clean_database=True)
        
        # Multiple related queries with message history
        message_history = []
        
        # First query
        result1 = rag.query_sync(
            "What are the main components of this codebase?",
            message_history
        )
        
        if result1["success"]:
            message_history.extend(result1["new_messages"])
            print("First query result:", result1["output"])
            
            # Follow-up query with context
            result2 = rag.query_sync(
                "Can you show me the implementation of the main class?",
                message_history
            )
            
            if result2["success"]:
                print("Follow-up query result:", result2["output"])
        
        # Export the graph
        rag.export_graph("final_graph.json")


if __name__ == "__main__":
    # Run examples (commented out for safety)
    # example_basic_usage()
    # example_optimization_usage()
    # example_advanced_usage()
    pass
