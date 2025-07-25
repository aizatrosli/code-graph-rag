import mimetypes
import shutil
import uuid
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from loguru import logger

from ..config import detect_provider_from_model, settings


class DocumentAnalyzer:
    """
    A tool to perform document analysis.
    Note: Multimodal analysis has been temporarily disabled in this version.
    """

    def __init__(self, project_root: str) -> None:
        self.project_root = Path(project_root).resolve()
        
        # Initialize client based on the orchestrator model's provider
        orchestrator_model = settings.active_orchestrator_model
        orchestrator_provider = detect_provider_from_model(orchestrator_model)

        if orchestrator_provider == "azure":
            try:
                deployment_name = orchestrator_model.replace("azure-", "")
                self.client = AzureChatOpenAI(
                    azure_deployment=deployment_name,
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                    api_key=settings.AZURE_OPENAI_API_KEY,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    temperature=0.3,
                )
                self.supports_vision = deployment_name in ["gpt-4o", "gpt-4o-mini", "gpt-4-vision-preview"]
            except Exception as e:
                logger.warning(f"Failed to initialize Azure OpenAI client: {e}")
                self.client = None
                self.supports_vision = False
        elif orchestrator_provider == "openai":
            try:
                self.client = ChatOpenAI(
                    model=orchestrator_model,
                    api_key=settings.OPENAI_API_KEY,
                    temperature=0.3,
                )
                self.supports_vision = orchestrator_model in ["gpt-4o", "gpt-4o-mini", "gpt-4-vision-preview"]
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.client = None
                self.supports_vision = False
        else:
            # Other providers (vLLM, local) don't support document analysis yet
            logger.warning(f"Document analysis not supported for provider: {orchestrator_provider}")
            self.client = None
            self.supports_vision = False

        logger.info(f"DocumentAnalyzer initialized with root: {self.project_root}")

    def analyze(self, file_path: str, question: str) -> str:
        """
        Analyzes a document with a specific question.
        Note: Currently supports only text-based analysis. 
        Vision capabilities are planned for future releases.
        """
        logger.info(
            f"[DocumentAnalyzer] Analyzing '{file_path}' with question: '{question}'"
        )
        
        if not self.client:
            return "Error: Document analysis is not available for the current LLM provider configuration."
        
        try:
            # Handle absolute paths by copying to .tmp folder
            if Path(file_path).is_absolute():
                source_path = Path(file_path)
                if not source_path.is_file():
                    return f"Error: File not found at '{file_path}'."

                # Create .tmp folder if it doesn't exist
                tmp_dir = self.project_root / ".tmp"
                tmp_dir.mkdir(exist_ok=True)

                # Copy file to .tmp with a unique filename to avoid collisions
                tmp_file = tmp_dir / f"{uuid.uuid4()}-{source_path.name}"
                shutil.copy2(source_path, tmp_file)
                full_path = tmp_file
                logger.info(f"Copied external file to: {full_path}")
            else:
                # Handle relative paths as before
                full_path = (self.project_root / file_path).resolve()
                full_path.relative_to(self.project_root)  # Security check

            if not full_path.is_file():
                return f"Error: File not found at '{file_path}'."

            # For now, we'll do simple text-based analysis
            # Full vision capabilities will be added in future releases
            try:
                # Try to read as text file first
                content = full_path.read_text(encoding='utf-8')
                
                # Create a simple prompt for text analysis
                from langchain_core.messages import HumanMessage, SystemMessage
                
                system_prompt = "You are a helpful assistant that analyzes documents and answers questions about their content."
                user_prompt = f"Document content:\n\n{content[:10000]}...\n\nQuestion: {question}\n\nPlease provide a detailed answer based on the document content."
                
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                response = self.client.invoke(messages)
                return response.content
                
            except UnicodeDecodeError:
                # If it's not a text file, return limitation message
                return f"Error: Currently only text-based document analysis is supported. Binary files like PDFs require vision capabilities that will be added in future releases."

        except ValueError as e:
            # Check if this is a security-related ValueError (from relative_to)
            if "does not start with" in str(e):
                err_msg = f"Security risk: Attempted to access file outside of project root: {file_path}"
                logger.error(err_msg)
                return f"Error: {err_msg}"
            else:
                # API-related ValueError
                logger.error(f"[DocumentAnalyzer] API validation error: {e}")
                return f"Error: API validation failed: {e}"
        except Exception as e:
            # Handle general errors
            logger.error(f"Document analysis error for '{file_path}': {e}")
            return f"Error: Failed to analyze document: {e}"
            return f"API error: {e}"
        except Exception as e:
            logger.error(
                f"Failed to analyze document '{file_path}': {e}", exc_info=True
            )
            return f"An error occurred during analysis: {e}"


def create_document_analyzer_tool(analyzer: DocumentAnalyzer):
    """Factory function to create the document analyzer tool."""

    @tool
    def analyze_document(
        file_path: Annotated[str, "The path to the document file (e.g., 'path/to/book.pdf')"],
        question: Annotated[str, "The specific question to ask about the document's content"]
    ) -> str:
        """
        Analyzes a document (like a PDF) to answer a specific question about its content.
        Use this tool when a user asks a question that requires understanding the content of a non-source-code file.

        Args:
            file_path: The path to the document file (e.g., 'path/to/book.pdf').
            question: The specific question to ask about the document's content.
        """
        try:
            result = analyzer.analyze(file_path, question)
            logger.debug(
                f"[analyze_document] Result type: {type(result)}, content: {result[:100] if result else 'None'}..."
            )
            return result
        except Exception as e:
            logger.error(
                f"[analyze_document] Exception during analysis: {e}", exc_info=True
            )
            if str(e).startswith("Error:") or str(e).startswith("API error:"):
                return str(e)
            return f"Error during document analysis: {e}"

    return analyze_document
