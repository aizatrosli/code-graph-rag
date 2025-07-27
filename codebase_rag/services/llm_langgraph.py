"""
LangGraph-based LLM service module with Azure OpenAI and vLLM support.
"""
import os
import re
import yake
import pandas as pd
from typing import cast

from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger
from ..prompts import (
    CYPHER_SYSTEM_PROMPT,
    LOCAL_CYPHER_SYSTEM_PROMPT,
    RAG_ORCHESTRATOR_SYSTEM_PROMPT,
)

ACRONYMDB_PATH = '/*/acronym.pkl/'
REPO_DESC = 'Related to LLM architecture'

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

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm
        self.system_prompt = LOCAL_CYPHER_SYSTEM_PROMPT

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
        

def get_orchestrator_system_prompt(input: str, user_req: str) -> str:
    """Get the system prompt for the orchestrator."""
    ndf = pd.DataFrame()
    if os.path.exists(ACRONYMDB_PATH):
        ndf = pd.read_pickle(ACRONYMDB_PATH)
        keys = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, dedupFunc='seqm', windowsSize=50, top=50, features=None).extract_keywords(input)
        keys = [key[0].lower() for key in keys]
        logger.debug(f"Keywords init: {keys}")
        keys = keys if keys else input.strip().split(' ')
        logger.debug(f"Keywords refine: {keys}")
        ndf = pd.concat([ndf[ndf['name'].str.lower() == d] for d in keys], ignore_index=True)
        ndf = ndf.drop_duplicates(['name', 'description'])
    if ndf.empty:
        return RAG_ORCHESTRATOR_SYSTEM_PROMPT.format(ACRONYM="", USER_REQ=user_req, GOAL=REPO_DESC)
    acronym = "**Acronym References:**\n"
    for i,a in enumerate(ndf.to_dict(orient='records')):
        acronym += f'{i}.  {a["name"]} - {a["description"]}\n'
    return RAG_ORCHESTRATOR_SYSTEM_PROMPT.format(ACRONYM=acronym, USER_REQ=user_req, GOAL=REPO_DESC)
