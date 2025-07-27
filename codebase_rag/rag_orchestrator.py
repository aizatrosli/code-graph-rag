"""
LangGraph workflow for the RAG orchestrator.
"""
from typing import Any, Dict, List
from loguru import logger

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.language_models.llms import BaseLLM
from langchain.schema.runnable.config import RunnableConfig
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse, get_client

from .services.llm_langgraph import get_orchestrator_system_prompt
from .config import CodebaseConfig

class RAGState(MessagesState):
    """State for the RAG workflow."""
    user_request: str = ""


def create_rag_workflow(tools: List[Any], settings: CodebaseConfig) -> StateGraph:
    """Create the LangGraph workflow for RAG orchestration."""
    
    # Get the LLM and bind tools to it
    llm = settings.ACTIVE_ORCHESTRATOR_MODEL
    llm_with_tools = llm.bind_tools(tools)
    
    # Create tool node
    tool_node = ToolNode(tools)
            
    def call_model(state: RAGState) -> Dict[str, Any]:
        """Call the model with the current state."""
        user_request = ""
        messages = state["messages"]
        # Add system message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            user_request = messages[-1].content if isinstance(messages[-1], HumanMessage) else state["user_request"]
            system_msg = SystemMessage(content=get_orchestrator_system_prompt(messages[-1].content, user_request))
            messages = [system_msg] + messages
        
        config = RunnableConfig(run_name=settings.LANGFUSE_PROJECT, recursion_limit=settings.RECURSION_LIMIT)
        if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
            Langfuse(host=settings.LANGFUSE_HOST, public_key=settings.LANGFUSE_PUBLIC_KEY, secret_key=settings.LANGFUSE_SECRET_KEY)
            lfcallback = CallbackHandler()
            config = RunnableConfig(callbacks=[lfcallback], run_name=settings.LANGFUSE_PROJECT, recursion_limit=settings.RECURSION_LIMIT)
        
        response = llm_with_tools.invoke(messages, config=config)
        return {"messages": [response], "user_request": user_request}
    
    def should_continue(state: RAGState) -> str:
        """Determine whether to continue with tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the last message has tool calls, continue with tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        # Otherwise, end
        return "end"
    
    # Build the graph
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": "__end__"
        }
    )
    
    # Always return to agent after using tools
    workflow.add_edge("tools", "agent")
    
    return workflow

class Response:
    def __init__(self, output: str, new_messages: List[BaseMessage]):
        self.output = output
        self._new_messages = new_messages
    
    def new_messages(self) -> List[BaseMessage]:
        """Return the new messages from this conversation turn."""
        return self._new_messages

class RAGOrchestrator:
    """LangGraph-based RAG orchestrator."""
    
    def __init__(self, tools: List[Any], settings: CodebaseConfig):
        self.workflow = create_rag_workflow(tools, settings)
        self.app = self.workflow.compile()
        logger.info("RAG orchestrator initialized with LangGraph")
    
    async def run(self, message: str, message_history: List[BaseMessage] = None) -> Any:
        """Run the orchestrator with a message."""
        if message_history is None:
            message_history = []
        
        # Add the new message to history
        messages = message_history + [HumanMessage(content=message)]
        
        # Run the workflow
        result = await self.app.ainvoke({"messages": messages})
        
        # Extract the final response
        final_messages = result["messages"]
        if final_messages:
            last_message = final_messages[-1]
            # Calculate new messages (everything after the original message history)
            new_messages = final_messages[len(message_history) + 1:]  # +1 to skip the user message we added
            
            return Response(last_message.content, new_messages)

        return Response("No response generated", [])
