# LangGraph Migration Summary with Azure OpenAI, vLLM, and DeepSeek

This document summarizes the migration from Pydantic AI to LangGraph with comprehensive LLM provider support including Azure OpenAI, vLLM, and DeepSeek in the code-graph-rag project.

## ✅ Completed Changes

### 1. Dependencies Updated
- **Removed**: 
  - `pydantic-ai-slim[google,openai,vertexai]>=0.2.18`
  - `langchain-google-genai>=2.0.8`
  - `langchain-google-vertexai>=2.0.12`
- **Added**: 
  - `langgraph>=0.2.66`
  - `langchain>=0.3.13`
  - `langchain-core>=0.3.28`
  - `langchain-openai>=0.2.14` (supports both OpenAI and Azure OpenAI)

### 2. LLM Provider Support
- **Azure OpenAI**: Primary provider with deployment-based model selection
- **DeepSeek**: Cost-effective AI provider with competitive performance
- **vLLM**: High-performance local inference server
- **OpenAI**: Direct OpenAI API support
- **Ollama**: Local model support via OpenAI-compatible API

### 3. Configuration Changes (config.py)

**New Provider Support:**
```python
def detect_provider_from_model(model: str) -> str:
    # Azure OpenAI (recommended)
    if "azure-" in model:
        return "azure"
    # DeepSeek (cost-effective)
    elif "deepseek-" in model:
        return "deepseek"
    # vLLM server support
    elif "vllm-" in model:
        return "vllm"
    # Direct OpenAI API
    elif model.startswith(("gpt-", "o1-", "text-")):
        return "openai"
    # Local models via Ollama
    else:
        return "local"
```

**Azure OpenAI Configuration:**
- Uses deployment names instead of direct model names
- Requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY
- Supports all Azure OpenAI models (GPT-4, GPT-3.5-turbo, etc.)

**DeepSeek Configuration:**
- Cost-effective alternative with competitive performance
- Requires DEEPSEEK_API_KEY
- Supports deepseek-chat, deepseek-coder, deepseek-v2
- Uses OpenAI-compatible API interface

**vLLM Configuration:**
- Connects to local vLLM server via OpenAI-compatible API
- Configurable base URL (default: http://localhost:8000)
- Supports any model deployed on vLLM

**Default Models:**
- Changed from `gemini-1.5-flash` to `azure-gpt-4o-mini`
- Cypher generation uses `azure-gpt-35-turbo` for efficiency

### 4. New LLM Service Module
- **Created**: `codebase_rag/services/llm_langgraph.py`
- **Renamed**: `codebase_rag/services/llm.py` → `codebase_rag/services/llm_old.py`
- **Features**:
  - `CypherGenerator` class for natural language to Cypher translation
  - `create_orchestrator_llm()` factory function
  - Support for Azure OpenAI, DeepSeek, vLLM, OpenAI, and Ollama providers
  - Provider-specific model initialization and configuration

### 5. New RAG Orchestrator
- **Created**: `codebase_rag/rag_orchestrator.py`
- **Features**:
  - LangGraph StateGraph workflow
  - Tool binding and execution
  - Message state management
  - Compatibility with existing pydantic_ai.run() interface

### 6. Tools Migration
All tools have been migrated from Pydantic AI `Tool` objects to LangChain `@tool` decorators:

- ✅ `codebase_query.py` - Knowledge graph querying
- ✅ `code_retrieval.py` - Code snippet retrieval
- ✅ `file_reader.py` - File reading
- ✅ `file_writer.py` - File creation
- ✅ `file_editor.py` - Code editing
- ✅ `directory_lister.py` - Directory listing
- ✅ `shell_command.py` - Shell command execution
- ✅ `document_analyzer.py` - Document analysis (migrated to Azure OpenAI)

### 7. Main Application Updates
- Updated imports in `main.py`
- Changed agent creation from `create_rag_orchestrator()` to `RAGOrchestrator()`
- Maintained compatibility with existing chat loop and optimization loop

## 🔧 Technical Changes

### Tool Function Signatures
**Before (Pydantic AI)**:
```python
def create_tool() -> Tool:
    async def tool_func(param: str) -> str:
        # implementation
    
    return Tool(
        function=tool_func,
        description="Tool description"
    )
```

**After (LangGraph)**:
```python
def create_tool():
    @tool
    async def tool_func(
        param: Annotated[str, "Parameter description"]
    ) -> str:
        """Tool description"""
        # implementation
    
    return tool_func
```

### LLM Integration
**Before**: Direct Pydantic AI model classes
**After**: LangChain chat model abstractions with provider-specific implementations

### Workflow Orchestration
**Before**: Pydantic AI Agent with tools
**After**: LangGraph StateGraph with ToolNode and conditional edges

## 🚀 Benefits

1. **Standardization**: Uses industry-standard LangChain ecosystem
2. **Flexibility**: LangGraph provides more control over agent workflows
3. **Community**: Larger community and more resources available
4. **Extensions**: Easier to integrate with other LangChain components
5. **Debugging**: Better debugging and visualization tools

## 🧪 Testing Status

- ✅ Basic imports and object creation work
- ✅ Tool creation and binding successful
- ✅ LangGraph workflow compiles
- ✅ Azure OpenAI provider detection and configuration
- ✅ DeepSeek provider detection and configuration
- ✅ vLLM provider support verified
- ⏳ End-to-end functionality testing pending (requires API keys)

## 🔄 API Compatibility

The migration maintains backward compatibility with the existing interface:
- `rag_agent.run(message, message_history)` still works
- Response object with `.output` attribute is preserved
- All tool functions maintain the same signatures

## 📝 Next Steps

1. **Testing**: Comprehensive testing with real Azure OpenAI and vLLM endpoints
2. **Vision Support**: Re-enable vision capabilities with Azure OpenAI
3. **Performance**: Compare performance across different providers
4. **Documentation**: Update user-facing documentation with new provider setup

## 🔍 Verification

To verify the migration works:

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.\.venv\Scripts\Activate.ps1  # Windows

# Test imports
python -c "from codebase_rag.rag_orchestrator import RAGOrchestrator; print('✅ Success')"

# Run the application (requires proper .env setup)
python -m codebase_rag.main start --repo-path /path/to/repo
```

The migration is now complete and ready for testing!
