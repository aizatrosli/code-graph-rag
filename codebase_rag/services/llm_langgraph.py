"""
LangGraph-based LLM service module.
"""
import os
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from loguru import logger

from ..config import detect_provider_from_model, settings
from ..prompts import (
    CYPHER_SYSTEM_PROMPT,
    LOCAL_CYPHER_SYSTEM_PROMPT,
    RAG_ORCHESTRATOR_SYSTEM_PROMPT,
)


class LLMGenerationError(Exception):
    """Custom exception for LLM generation failures."""
    pass


def _clean_cypher_response(response_text: str) -> str:
    """Utility to clean up common LLM formatting artifacts from a Cypher query."""
    query = response_text.strip().replace("`", "")
    if query.startswith("cypher"):
        query = query[6:].strip()
    if not query.endswith(";"):
        query += ";"
    return query


class CypherGenerator:
    """Generates Cypher queries from natural language."""

    def __init__(self) -> None:
        try:
            # Get active cypher model and detect its provider
            cypher_model_id = settings.active_cypher_model
            cypher_provider = detect_provider_from_model(cypher_model_id)

            # Configure model based on detected provider
            if cypher_provider == "gemini":
                if settings.GEMINI_PROVIDER == "vertex":
                    self.llm = ChatVertexAI(
                        model=cypher_model_id,
                        project=settings.GCP_PROJECT_ID,
                        location=settings.GCP_REGION,
                        credentials=settings.GCP_SERVICE_ACCOUNT_FILE,
                        temperature=0.1,
                    )
                else:
                    self.llm = ChatGoogleGenerativeAI(
                        model=cypher_model_id,
                        google_api_key=settings.GEMINI_API_KEY,
                        temperature=0.1,
                    )
                self.system_prompt = CYPHER_SYSTEM_PROMPT

            elif cypher_provider == "openai":
                self.llm = ChatOpenAI(
                    model=cypher_model_id,
                    api_key=settings.OPENAI_API_KEY,
                    temperature=0.1,
                )
                self.system_prompt = CYPHER_SYSTEM_PROMPT

            else:  # local
                self.llm = ChatOpenAI(
                    model=cypher_model_id,
                    api_key=settings.LOCAL_MODEL_API_KEY,
                    base_url=str(settings.LOCAL_MODEL_ENDPOINT),
                    temperature=0.1,
                )
                self.system_prompt = LOCAL_CYPHER_SYSTEM_PROMPT

        except Exception as e:
            raise LLMGenerationError(
                f"Failed to initialize CypherGenerator: {e}"
            ) from e

    async def generate(self, natural_language_query: str) -> str:
        """Generate Cypher query from natural language."""
        logger.info(
            f"  [CypherGenerator] Generating query for: '{natural_language_query}'"
        )
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=natural_language_query),
            ]
            
            result = await self.llm.ainvoke(messages)
            
            if not isinstance(result, AIMessage) or not result.content:
                raise LLMGenerationError(
                    f"LLM did not generate a valid response. Output: {result}"
                )
            
            # Validate that the response contains a Cypher query
            output = result.content
            if (
                not isinstance(output, str)
                or len(output.strip()) < 5
                or "MATCH" not in output.upper()
            ):
                raise LLMGenerationError(
                    f"LLM did not generate a valid query. Output: {output}"
                )

            query = _clean_cypher_response(output)
            logger.info(f"  [CypherGenerator] Generated Cypher: {query}")
            return query

        except Exception as e:
            logger.error(f"  [CypherGenerator] Error: {e}")
            raise LLMGenerationError(f"Cypher generation failed: {e}") from e


def create_orchestrator_llm() -> BaseChatModel:
    """Factory function to create the main RAG orchestrator LLM."""
    try:
        # Get active orchestrator model and detect its provider
        orchestrator_model_id = settings.active_orchestrator_model
        orchestrator_provider = detect_provider_from_model(orchestrator_model_id)

        if orchestrator_provider == "gemini":
            if settings.GEMINI_PROVIDER == "vertex":
                llm = ChatVertexAI(
                    model=orchestrator_model_id,
                    project=settings.GCP_PROJECT_ID,
                    location=settings.GCP_REGION,
                    credentials=settings.GCP_SERVICE_ACCOUNT_FILE,
                    temperature=0.7,
                )
            else:
                llm = ChatGoogleGenerativeAI(
                    model=orchestrator_model_id,
                    google_api_key=settings.GEMINI_API_KEY,
                    temperature=0.7,
                )

        elif orchestrator_provider == "local":
            llm = ChatOpenAI(
                model=orchestrator_model_id,
                api_key=settings.LOCAL_MODEL_API_KEY,
                base_url=str(settings.LOCAL_MODEL_ENDPOINT),
                temperature=0.7,
            )

        else:  # openai provider
            llm = ChatOpenAI(
                model=orchestrator_model_id,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.7,
            )

        return llm

    except Exception as e:
        raise LLMGenerationError(
            f"Failed to initialize orchestrator LLM: {e}"
        ) from e


def get_orchestrator_system_prompt() -> str:
    """Get the system prompt for the orchestrator."""
    return RAG_ORCHESTRATOR_SYSTEM_PROMPT
