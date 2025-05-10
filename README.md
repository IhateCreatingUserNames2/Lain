# ADK Orchestrator Agent with MemoryBlossom and A2A Wrapper

This project implements an intelligent agent using the Google Agent Development Kit (ADK).
The agent features an advanced memory system called "MemoryBlossom" and is exposed
for interaction via the Agent2Agent (A2A) protocol through a FastAPI wrapper.
This allows it to be registered with an AIRA Hub and accessed by MCP clients.

## Project Structure

-   **`memory_system/`**: Contains the implementation of the MemoryBlossom system.
    -   `memory_models.py`: Defines the `Memory` data class.
    -   `embedding_utils.py`: Utilities for generating and comparing text embeddings.
    -   `memory_blossom.py`: The core `MemoryBlossom` class managing different memory types.
    -   `memory_connector.py`: Class for analyzing and creating connections between memories.
-   **`orchestrator_adk_agent.py`**: Defines the main ADK `LlmAgent`, its tools for interacting with MemoryBlossom, and the ADK `Runner`.
-   **`a2a_wrapper/`**: Contains the FastAPI server that exposes the ADK agent via A2A.
    -   `main.py`: The FastAPI application logic.
    -   `models.py`: Pydantic models for the A2A protocol.
-   **`aira_hub_config.py`**: An *illustrative* configuration showing how this A2A-wrapped agent might be registered and understood by an AIRA Hub. This file is not run directly by this project.
-   `requirements.txt`: Python dependencies.
-   `README.md`: This file.

## Features

-   **ADK Agent**: An `LlmAgent` that can converse and utilize tools.
-   **MemoryBlossom**:
    -   Supports multiple memory types (Explicit, Emotional, Procedural, etc.).
    -   Uses different `sentence-transformers` models for embedding various memory types.
    -   Calculates memory salience, applies decay.
    -   Connects related memories using `MemoryConnector`.
    -   Persists memories to a JSON file (`memory_blossom_data.json`).
-   **A2A Wrapper**:
    -   Exposes the ADK agent via a `/.well-known/agent.json` Agent Card.
    -   Handles A2A `tasks/send` requests through a JSON-RPC endpoint.
    -   Manages ADK sessions based on A2A Task IDs.

## Setup

1.  **Create a Python Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **API Keys:**
    -   Ensure your `GOOGLE_API_KEY` environment variable is set if you are using Gemini models directly with ADK.
    -   If using other models via LiteLLM (e.g., OpenAI's GPT), set the corresponding API key (e.g., `OPENAI_API_KEY`). The `orchestrator_adk_agent.py` can be modified to use `LiteLlm`.
    -   You can set these in your shell or create a `.env` file in the project root and load it (e.g., using `python-dotenv` in your scripts, though not explicitly added in these examples for brevity).

## Running the System

You need to run the A2A Wrapper Server. The ADK agent and MemoryBlossom are part of this server process.

1.  **Start the A2A Wrapper Server:**
    Navigate to the project root directory.
    ```bash
    python -m uvicorn a2a_wrapper.main:app --host 0.0.0.0 --port 8090 --reload
    ```
    -   The server will start, typically on `http://localhost:8090`.
    -   The Agent Card will be available at `http://localhost:8090/.well-known/agent.json`.
    -   The A2A JSON-RPC endpoint will be `http://localhost:8090/`.

2.  **Register with AIRA Hub (Manual Step):**
    -   Start your AIRA Hub application.
    -   Send a POST request to your AIRA Hub's `/register` endpoint. The body should be a JSON payload describing this A2A agent. See `aira_hub_config.py` for an example structure.
        -   **Key fields for registration:**
            -   `url`: The URL of this A2A Wrapper Server (e.g., `http://localhost:8090`).
            -   `type`: "a2a_bridged" (or similar, depending on your Hub's convention).
            -   `a2a_agent_card_url`: `http://localhost:8090/.well-known/agent.json`.
            -   The Hub will then fetch the card to understand the A2A agent's skills and generate corresponding MCP tool definitions.

3.  **Interact via MCP Client:**
    -   Use your MCP client (e.g., `simpleStreamableHttp.ts`) to connect to your AIRA Hub.
    -   List tools: You should see a tool derived from the A2A agent's "general_conversation" skill (e.g., named "TalkToMemoryBlossomADK").
    -   Call the tool with appropriate arguments. For the "general_conversation" skill, the MCP tool call would look something like:
        `call-tool TalkToMemoryBlossomADK '{"user_input": "Hello, can you remember that I like hiking?"}'`

## How it Works (Flow for an MCP Client)

1.  **MCP Client** -> **AIRA Hub**: `list_tools`
2.  **AIRA Hub**:
    -   Identifies registered A2A-bridged agents.
    -   Fetches `agent.json` from the A2A Wrapper (`http://localhost:8090/.well-known/agent.json`).
    -   Translates the "general_conversation" A2A skill into an MCP tool definition.
3.  **AIRA Hub** -> **MCP Client**: Returns list of available tools, including one for interacting with this ADK agent.
4.  **MCP Client** -> **AIRA Hub**: `call_tool` (e.g., `TalkToMemoryBlossomADK` with `{"user_input": "..."}`)
5.  **AIRA Hub**:
    -   Receives MCP `call_tool`.
    -   Identifies it's for the A2A-bridged ADK agent.
    -   Constructs an A2A `tasks/send` JSON-RPC request.
        -   `method`: "tasks/send"
        -   `params.id`: New A2A Task ID (can be MCP call ID).
        -   `params.message.parts[0].type`: "data"
        -   `params.message.parts[0].data`: `{"skill_id": "general_conversation", "user_input": "..."}` (or however your Hub bridges MCP args to A2A skill params).
6.  **AIRA Hub** -> **A2A Wrapper Server** (`http://localhost:8090/`): Sends A2A `tasks/send` request.
7.  **A2A Wrapper Server**:
    -   Receives A2A request.
    -   Extracts `user_input`.
    -   Gets/creates an ADK session ID mapped to the A2A Task ID.
    -   Updates conversation history in ADK session state.
    -   Calls `adk_runner.run_async()` with the `user_input`.
8.  **ADK Orchestrator Agent**:
    -   Processes the input.
    -   May use `add_memory_tool_func` or `recall_memories_tool_func` (interacting with `MemoryBlossom`).
    -   Generates a final text response.
9.  **A2A Wrapper Server**:
    -   Receives final ADK response.
    -   Updates ADK session history with agent response.
    -   Formats it as an A2A `TaskResult` (with an artifact containing the text).
    -   Sends A2A JSON-RPC response back to AIRA Hub.
10. **AIRA Hub**:
    -   Receives A2A `TaskResult`.
    -   Extracts the text artifact.
    -   Translates it into an MCP tool response (e.g., `{"text_response": "..."}`).
11. **AIRA Hub** -> **MCP Client**: Sends MCP tool response.

## Standalone ADK Agent Testing

You can test the ADK agent and MemoryBlossom system directly by running:
```bash
python orchestrator_adk_agent.py