# orchestrator_adk_agent.py
import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


from dotenv import load_dotenv
# Assuming .env is in the same directory or project root
# If orchestrator_adk_agent.py is in the root, this is fine:
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if not os.path.exists(dotenv_path): # Fallback if it's one level up (e.g. running from a subfolder)
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')

if os.path.exists(dotenv_path):
    print(f"Orchestrator ADK: Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    print(f"Orchestrator ADK: .env file not found. Relying on environment variables.")
# +++++++++++++++++++++++++++++++++++++++++++++


from google.adk.agents import LlmAgent
# Import LiteLlm model wrapper from ADK
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool, ToolContext
from google.adk.sessions import InMemorySessionService # Or DatabaseSessionService
from google.adk.runners import Runner
from google.genai.types import Content as ADKContent, Part as ADKPart

# Import from memory_system
from memory_system.memory_blossom import MemoryBlossom
from memory_system.memory_connector import MemoryConnector
from memory_system.memory_models import Memory



# --- Configuration ---
# Make sure OPENROUTER_API_KEY is set in your environment!
# e.g., export OPENROUTER_API_KEY="sk-or-v1-..."
# LiteLLM will pick it up automatically.

# Optional OpenRouter headers for analytics/leaderboards
os.environ["OR_SITE_URL"] = "https://3e7d-189-28-2-52.ngrok-free.app" # Your site
os.environ["OR_APP_NAME"] = "Lain"    # Your app name

# --- CHOOSE YOUR OPENROUTER MODEL ---
# Option 1: Specify a model explicitly via OpenRouter
# Find model slugs on OpenRouter's website (e.g., Models page)
# AGENT_MODEL_STRING = "openrouter/google/gemini-flash-1.5"
# AGENT_MODEL_STRING = "openrouter/openai/gpt-4o-mini"
AGENT_MODEL_STRING = "openrouter/openai/gpt-4o-mini" # Example: Haiku

# Option 2: Use OpenRouter's "auto" router (NotDiamond)
# AGENT_MODEL_STRING = "openrouter/auto"

# Option 3: Use OpenRouter with web search (for supported models)
# AGENT_MODEL_STRING = "openrouter/openai/gpt-4o:online" # Note the :online suffix

AGENT_MODEL = LiteLlm(model=AGENT_MODEL_STRING) # Pass the OpenRouter model string to ADK's LiteLlm

ADK_APP_NAME = "OrchestratorMemoryApp_OpenRouter"

# --- Initialize MemoryBlossom (remains the same) ---
memory_blossom_persistence_file = os.getenv("MEMORY_BLOSSOM_PERSISTENCE_PATH", "memory_blossom_data.json")
memory_blossom_instance = MemoryBlossom(persistence_path=memory_blossom_persistence_file)
memory_connector_instance = MemoryConnector(memory_blossom_instance)
memory_blossom_instance.set_memory_connector(memory_connector_instance)

# --- ADK Tools for MemoryBlossom (remain the same) ---
def add_memory_tool_func(
    content: str,
    memory_type: str,
    emotion_score: float = 0.0,
    coherence_score: float = 0.5,
    novelty_score: float = 0.5,
    initial_salience: float = 0.5,
    metadata_json: Optional[str] = None,
    tool_context: Optional[ToolContext] = None
) -> Dict[str, Any]:
    """
    Adds a memory to the MemoryBlossom system.
    Args:
        content: The textual content of the memory.
        memory_type: The type of memory (e.g., Explicit, Emotional, Procedural, Flashbulb, Somatic, Liminal, Generative).
        emotion_score: Emotional intensity of the memory (0.0 to 1.0).
        coherence_score: How well-structured or logical the memory is (0.0 to 1.0).
        novelty_score: How unique or surprising the memory is (0.0 to 1.0).
        initial_salience: Initial importance of the memory (0.0 to 1.0).
        metadata_json: Optional JSON string representing a dictionary of additional metadata.
    """
    print(f"--- TOOL: add_memory_tool_func called with type: {memory_type} ---")
    parsed_metadata = None
    if metadata_json:
        try:
            parsed_metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid JSON format for metadata."}
    try:
        memory = memory_blossom_instance.add_memory(
            content=content,
            memory_type=memory_type,
            metadata=parsed_metadata,
            emotion_score=emotion_score,
            coherence_score=coherence_score,
            novelty_score=novelty_score,
            initial_salience=initial_salience
        )
        memory_blossom_instance.save_memories()
        return {"status": "success", "memory_id": memory.id, "message": f"Memory of type '{memory_type}' added."}
    except Exception as e:
        print(f"Error in add_memory_tool_func: {str(e)}")
        return {"status": "error", "message": str(e)}

def recall_memories_tool_func(
    query: str,
    target_memory_types_json: Optional[str] = None,
    top_k: int = 5,
    tool_context: Optional[ToolContext] = None
) -> Dict[str, Any]:
    """
    Recalls memories from the MemoryBlossom system based on a query.
    Args:
        query: The search query to find relevant memories.
        target_memory_types_json: Optional JSON string of a list of memory types to specifically search within.
        top_k: The maximum number of memories to return.
    """
    print(f"--- TOOL: recall_memories_tool_func called with query: {query[:30]}... ---")
    target_types_list: Optional[List[str]] = None
    if target_memory_types_json:
        try:
            target_types_list = json.loads(target_memory_types_json)
            if not isinstance(target_types_list, list) or not all(isinstance(item, str) for item in target_types_list):
                return {"status": "error", "message": "target_memory_types_json must be a JSON string of a list of strings."}
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid JSON format for target_memory_types_json."}
    try:
        conversation_history = None
        if tool_context and tool_context.state:
            conversation_history = tool_context.state.get('conversation_history', [])

        recalled_memories = memory_blossom_instance.retrieve_memories(
            query=query,
            target_memory_types=target_types_list,
            top_k=top_k,
            conversation_context=conversation_history
        )
        return {
            "status": "success",
            "count": len(recalled_memories),
            "memories": [mem.to_dict() for mem in recalled_memories]
        }
    except Exception as e:
        print(f"Error in recall_memories_tool_func: {str(e)}")
        return {"status": "error", "message": str(e)}

add_memory_adk_tool = FunctionTool(func=add_memory_tool_func)
recall_memories_adk_tool = FunctionTool(func=recall_memories_tool_func)

# --- Orchestrator ADK Agent Definition (instruction largely same) ---
orchestrator_agent_instruction = """
You are the Orchestrator Agent, powered by a sophisticated MemoryBlossom system.
Your primary role is to converse with the user, understand their needs, and intelligently interact with your memory.

Memory Interaction:
- When the user provides information they want you to remember, or when you deem a piece of information from the conversation important to store, use the 'add_memory_tool_func'.
  - You MUST specify the 'memory_type'. Choose from: Explicit, Emotional, Procedural, Flashbulb, Somatic, Liminal, Generative.
  - Briefly explain your choice of memory_type to the user when you confirm storage.
  - You can optionally provide 'emotion_score', 'coherence_score', 'novelty_score', 'initial_salience', and 'metadata_json'.
- When the user asks a question or discusses something that might relate to past information, use the 'recall_memories_tool_func' to search your memory.
  - You can optionally specify 'target_memory_types_json' (a JSON list of strings) to focus your search.

Conversational Style:
- Be natural, engaging, and context-aware.
- Acknowledge when you are storing or recalling memories.
- If you recall memories, incorporate them naturally into your response. Don't just list them.
- If MemoryBlossom returns an error or no memories, inform the user gracefully.
"""

# If you want structured output from the agent (and the OpenRouter model supports it)
# from pydantic import BaseModel, Field
# class AgentFinalResponseSchema(BaseModel):
#     responseText: str = Field(description="The agent's final textual response to the user.")
#     memoryActionTaken: Optional[str] = Field(None, description="Description of memory action, e.g., 'Stored memory X' or 'Recalled Y memories'.")

orchestrator_adk_agent = LlmAgent(
    name="MemoryBlossomOrchestratorOpenRouter",
    model=AGENT_MODEL, # Key change: Use the LiteLlm wrapped OpenRouter model
    instruction=orchestrator_agent_instruction,
    tools=[add_memory_adk_tool, recall_memories_adk_tool],
    # output_schema=AgentFinalResponseSchema # Uncomment if using structured output
)

# --- ADK Runner and Session Service (remains same) ---
adk_session_service = InMemorySessionService()
adk_runner = Runner(
    agent=orchestrator_adk_agent,
    app_name=ADK_APP_NAME,
    session_service=adk_session_service
)

print(f"ADK Orchestrator Agent '{orchestrator_adk_agent.name}' and Runner initialized with LiteLLM model '{AGENT_MODEL_STRING}'.")
print(f"MemoryBlossom instance ready. Loaded {sum(len(m) for m in memory_blossom_instance.memory_stores.values())} memories from persistence.")

# --- Standalone Test Function (remains same, but will use OpenRouter now) ---
async def run_adk_test_conversation():
    user_id = "test_user_adk_openrouter"
    session_id = "test_session_adk_openrouter"

    _ = adk_session_service.get_session(app_name=ADK_APP_NAME, user_id=user_id, session_id=session_id) or \
        adk_session_service.create_session(app_name=ADK_APP_NAME, user_id=user_id, session_id=session_id, state={'conversation_history': []})

    queries = [
        "Hello! I'm exploring how AI can manage different types of memories.",
        "Please remember that my research focus is 'emergent narrative structures' and I find 'liminal spaces' fascinating. Store this as an Explicit memory, with high novelty.",
        "What is my research focus and what do I find fascinating?",
        "Let's try storing an emotional memory: 'The sunset over the mountains was breathtakingly beautiful, a truly awe-inspiring moment.' Set emotion_score to 0.9.",
        "What beautiful moment did I describe?",
        "Can you search for memories related to 'sunsets' or 'mountains' in my Emotional memories? (target_memory_types_json='[\"Emotional\"]')",
        "Thank you, that's all for now."
    ]
    current_adk_session_state = adk_runner.session_service.get_session(ADK_APP_NAME, user_id, session_id).state
    if 'conversation_history' not in current_adk_session_state:
        current_adk_session_state['conversation_history'] = []

    for query in queries:
        print(f"\nUSER: {query}")
        current_adk_session_state['conversation_history'].append({"role": "user", "content": query})
        adk_input_content = ADKContent(role="user", parts=[ADKPart(text=query)])
        final_response_text = "ADK Agent: (No final text response)"
        async for event in adk_runner.run_async(
            user_id=user_id, session_id=session_id, new_message=adk_input_content
        ):
            if event.is_final_response():
                if event.content and event.content.parts and event.content.parts[0].text:
                    final_response_text = event.content.parts[0].text.strip()
                break
        print(f"AGENT: {final_response_text}")
        current_adk_session_state['conversation_history'].append({"role": "assistant", "content": final_response_text})
        current_adk_session_state = adk_runner.session_service.get_session(ADK_APP_NAME, user_id, session_id).state


if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: OPENROUTER_API_KEY environment variable is not set.             !!!")
        print("!!! The agent will likely fail to make LLM calls.                          !!!")
        print("!!! Please set it before running. e.g., export OPENROUTER_API_KEY='sk-or-...' !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # asyncio.run(run_adk_test_conversation())
    print("Orchestrator ADK Agent (using OpenRouter via LiteLLM) and MemoryBlossom are set up.")
    print("To test, uncomment the asyncio.run line or use the A2A wrapper.")