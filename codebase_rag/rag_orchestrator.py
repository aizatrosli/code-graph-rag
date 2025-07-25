"""
LangGraph workflow for the RAG orchestrator.
"""
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from loguru import logger

from .services.llm_langgraph import create_orchestrator_llm, get_orchestrator_system_prompt


class RAGState(MessagesState):
    """State for the RAG workflow."""
    pass


def create_rag_workflow(tools: List[Any]) -> StateGraph:
    """Create the LangGraph workflow for RAG orchestration."""
    
    # Get the LLM and bind tools to it
    llm = create_orchestrator_llm()
    llm_with_tools = llm.bind_tools(tools)
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    def call_model(state: RAGState) -> Dict[str, Any]:
        """Call the model with the current state."""
        messages = state["messages"]
        
        # Add system message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            system_msg = SystemMessage(content=get_orchestrator_system_prompt())
            messages = [system_msg] + messages
        
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
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


class RAGOrchestrator:
    """LangGraph-based RAG orchestrator."""
    
    def __init__(self, tools: List[Any]):
        self.workflow = create_rag_workflow(tools)
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
            # Create a simple response object that mimics pydantic_ai's response
            class Response:
                def __init__(self, output: str):
                    self.output = output
            
            return Response(last_message.content)
        
        return Response("No response generated")
