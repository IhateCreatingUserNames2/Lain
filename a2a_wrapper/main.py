# a2a_wrapper/main.py
import uvicorn
from fastapi import FastAPI, Request as FastAPIRequest, HTTPException
from fastapi.responses import JSONResponse
import json
import uuid
from datetime import datetime, timezone
from typing import Union, Dict, Any
import logging
import os
from dotenv import load_dotenv

# --- Load .env file (assuming .env is in E:\ProjetosPython\Lain) ---
# This path calculation assumes main.py is in a subfolder of the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(PROJECT_ROOT, '.env')

if os.path.exists(dotenv_path):
    print(f"A2A Wrapper: Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    print(f"A2A Wrapper: .env file not found at {dotenv_path}. Relying on environment variables.")

# --- Module Imports (Now absolute from project root where orchestrator_adk_agent.py lives) ---
# orchestrator_adk_agent.py is in the PROJECT_ROOT (E:\ProjetosPython\Lain)
from orchestrator_adk_agent import adk_runner, orchestrator_adk_agent, ADK_APP_NAME

# models.py is inside the a2a_wrapper package
from a2a_wrapper.models import (
    A2APart, A2AMessage, A2ATaskSendParams, A2AArtifact,
    A2ATaskStatus, A2ATaskResult, A2AJsonRpcRequest, A2AJsonRpcResponse,
    AgentCard, AgentCardSkill, AgentCardProvider, AgentCardAuthentication, AgentCardCapabilities
)

# ADK types
from google.genai.types import Content as ADKContent
from google.genai.types import Part as ADKPart
from google.adk.sessions import Session as ADKSession, logger

# --- Configuration for this A2A Wrapper Server ---
A2A_WRAPPER_HOST = os.getenv("A2A_WRAPPER_HOST", "0.0.0.0")
A2A_WRAPPER_PORT = int(os.getenv("A2A_WRAPPER_PORT", "8090"))
A2A_WRAPPER_BASE_URL = os.getenv("A2A_WRAPPER_BASE_URL", f"http://localhost:{A2A_WRAPPER_PORT}")

app = FastAPI(
    title="Orchestrator Agent A2A Wrapper",
    description="Exposes the MemoryBlossom Orchestrator ADK agent via the A2A protocol."
)

# --- Agent Card Definition (remains the same) ---
AGENT_CARD_DATA = AgentCard(
    name="Lain",
    # ... (rest of AgentCard definition as before) ...
    description="A conversational agent with advanced memory capabilities (MemoryBlossom), "
                "accessible via A2A. It can store and recall information of various types.",
    url=A2A_WRAPPER_BASE_URL, # This server's URL
    version="1.0.0",
    provider=AgentCardProvider(organization="LocalDev", url="https://3e7d-189-28-2-52.ngrok-free.app"),
    capabilities=AgentCardCapabilities(streaming=False, pushNotifications=False),
    authentication=AgentCardAuthentication(schemes=[]),
    skills=[
        AgentCardSkill(
            id="general_conversation",
            name="General Conversation & Memory Interaction",
            description="Engage in a conversation. The agent can use its MemoryBlossom system "
                        "to store or recall information based on the conversation. "
                        "You can ask it to remember things or query its memory.",
            tags=["chat", "conversation", "memory", "adk"],
            examples=[
                "Remember that I like chocolate ice cream.",
                "What was the project I mentioned I was working on yesterday?",
                "Store this as an Emotional memory: 'Winning the competition was exhilarating!'",
                "What memories do you have related to 'quantum physics'?"
            ],
            parameters={
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "The textual input from the user for the conversation."
                    },
                    "a2a_task_id_override": {
                        "type": "string",
                        "description": "Optional: Override the A2A task ID for session mapping.",
                        "nullable": True
                    }
                },
                "required": ["user_input"]
            }
        )
    ]
)

@app.get("/.well-known/agent.json", response_model=AgentCard, response_model_exclude_none=True)
async def get_agent_card():
    return AGENT_CARD_DATA

a2a_task_to_adk_session_map: Dict[str, str] = {}
adk_session_store: Dict[str, ADKSession] = {}

@app.post("/", response_model=A2AJsonRpcResponse, response_model_exclude_none=True)
async def handle_a2a_rpc(rpc_request: A2AJsonRpcRequest, http_request: FastAPIRequest):
    # ... (rest of the handle_a2a_rpc function remains the same as provided previously) ...
    client_host = http_request.client.host if http_request.client else "unknown"
    print(f"\nA2A Wrapper: Received request from {client_host}: Method={rpc_request.method}, RPC_ID={rpc_request.id}")

    if rpc_request.method == "tasks/send":
        if rpc_request.params is None:
            print("A2A Wrapper: Error - Missing 'params' for tasks/send")
            return A2AJsonRpcResponse(id=rpc_request.id, error={"code": -32602, "message": "Invalid params: missing"})

        try:
            task_params = rpc_request.params
            print(f"A2A Wrapper: Processing tasks/send for A2A Task ID: {task_params.id}")

            user_utterance = ""
            if task_params.message and task_params.message.parts:
                first_part = task_params.message.parts[0]
                if first_part.type == "data" and first_part.data and "user_input" in first_part.data:
                    user_utterance = first_part.data["user_input"]
                    print(f"A2A Wrapper: Extracted user_input from DataPart: '{user_utterance[:50]}...'")
                elif first_part.type == "text" and first_part.text:
                    user_utterance = first_part.text
                    print(f"A2A Wrapper: Extracted user_input from TextPart: '{user_utterance[:50]}...'")

            if not user_utterance:
                print("A2A Wrapper: Error - No user_input found in A2A message.")
                return A2AJsonRpcResponse(id=rpc_request.id, error={"code": -32602, "message": "Invalid params: user_input missing"})

            a2a_current_task_id = task_params.id
            if task_params.message.parts[0].data and task_params.message.parts[0].data.get("a2a_task_id_override"):
                 a2a_current_task_id = task_params.message.parts[0].data["a2a_task_id_override"]
                 print(f"A2A Wrapper: Using overridden A2A Task ID for session mapping: {a2a_current_task_id}")

            adk_session_id = a2a_task_to_adk_session_map.get(a2a_current_task_id)
            adk_user_id = f"a2a_user_for_task_{a2a_current_task_id}"

            if not adk_session_id:
                adk_session_id = f"adk_session_for_{a2a_current_task_id}_{str(uuid.uuid4())[:8]}"
                a2a_task_to_adk_session_map[a2a_current_task_id] = adk_session_id
                current_session = adk_runner.session_service.create_session(
                    app_name=ADK_APP_NAME,
                    user_id=adk_user_id,
                    session_id=adk_session_id,
                    state={'conversation_history': []}
                )
                adk_session_store[adk_session_id] = current_session
                print(f"A2A Wrapper: Created new ADK session '{adk_session_id}' for A2A Task '{a2a_current_task_id}'")
            else:
                current_session = adk_runner.session_service.get_session(
                    app_name=ADK_APP_NAME, user_id=adk_user_id, session_id=adk_session_id
                )
                if not current_session:
                    current_session = adk_runner.session_service.create_session(
                        app_name=ADK_APP_NAME, user_id=adk_user_id, session_id=adk_session_id, state={'conversation_history': []}
                    )
                adk_session_store[adk_session_id] = current_session
                print(f"A2A Wrapper: Reusing ADK session '{adk_session_id}' for A2A Task '{a2a_current_task_id}'")

            if 'conversation_history' not in current_session.state:
                current_session.state['conversation_history'] = []
            current_session.state['conversation_history'].append({"role": "user", "content": user_utterance})

            adk_input_content = ADKContent(role="user", parts=[ADKPart(text=user_utterance)])
            print(f"A2A Wrapper: Running ADK agent for session '{adk_session_id}' with input: '{user_utterance[:100]}...'")
            adk_agent_final_text_response = None

            async for event in adk_runner.run_async(
                user_id=adk_user_id,
                session_id=adk_session_id,
                new_message=adk_input_content
            ):
                print(f"  ADK Event: Author={event.author}, Final={event.is_final_response()}, Content Present={bool(event.content)}")
                if event.get_function_calls():
                    fc = event.get_function_calls()[0]
                    print(f"    ADK FunctionCall by {event.author}: {fc.name}({fc.args})")
                if event.get_function_responses():
                    fr = event.get_function_responses()[0]
                    print(f"    ADK FunctionResponse to {event.author}: {fr.name} -> {str(fr.response)[:100]}...")
                if event.is_final_response():
                    if event.content and event.content.parts and event.content.parts[0].text:
                        adk_agent_final_text_response = event.content.parts[0].text.strip()
                        print(f"  ADK Final Response Text: '{adk_agent_final_text_response[:100]}...'")
                    break
            if adk_agent_final_text_response is None:
                adk_agent_final_text_response = "(ADK agent did not provide a textual response for this turn)"
            current_session.state['conversation_history'].append({"role": "assistant", "content": adk_agent_final_text_response})

            a2a_response_artifact = A2AArtifact(
                parts=[A2APart(type="text", text=adk_agent_final_text_response)]
            )
            a2a_task_status = A2ATaskStatus(state="completed")
            a2a_task_result = A2ATaskResult(
                id=task_params.id,
                sessionId=task_params.sessionId,
                status=a2a_task_status,
                artifacts=[a2a_response_artifact]
            )
            print(f"A2A Wrapper: Sending A2A response for Task ID {task_params.id}")
            return A2AJsonRpcResponse(id=rpc_request.id, result=a2a_task_result)

        except ValueError as ve:
            print(f"A2A Wrapper: Value Error processing tasks/send: {ve}")
            return A2AJsonRpcResponse(id=rpc_request.id, error={"code": -32602, "message": f"Invalid params: {str(ve)}"})
        except Exception as e:
            # Use the logger for error messages with traceback
            if 'logger' in globals() or 'logger' in locals():
                logger.error(f"A2A Wrapper: Internal Error processing tasks/send: {e}", exc_info=True)
            else:  # Fallback if logger wasn't set up in this file
                print(f"A2A Wrapper: Internal Error processing tasks/send: {e}")
                import traceback
                traceback.print_exc()  # Manually print traceback if no logger

            return A2AJsonRpcResponse(id=rpc_request.id,
                                      error={"code": -32000, "message": f"Internal Server Error: {str(e)}"})
    else:
        print(f"A2A Wrapper: Method '{rpc_request.method}' not supported.")
        return A2AJsonRpcResponse(id=rpc_request.id, error={"code": -32601, "message": f"Method not found: {rpc_request.method}"})


if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: OPENROUTER_API_KEY is not set in environment or .env file.    !!!")
        print("!!! The agent will likely fail to make LLM calls via OpenRouter.           !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    print(f"Starting ADK A2A Wrapper Server on {A2A_WRAPPER_HOST}:{A2A_WRAPPER_PORT}")
    print(f"Agent Card will be available at: {A2A_WRAPPER_BASE_URL}/.well-known/agent.json")
    # Ensure this runs from the project root (Lain) for uvicorn to find 'a2a_wrapper.main'
    # Command to run from E:\ProjetosPython\Lain:
    # python -m uvicorn a2a_wrapper.main:app --host 0.0.0.0 --port 8090 --reload
    uvicorn.run("main:app", host=A2A_WRAPPER_HOST, port=A2A_WRAPPER_PORT, reload=True, app_dir="a2a_wrapper")