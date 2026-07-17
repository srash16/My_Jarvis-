import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentType, initialize_agent
from langchain_classic.memory import ConversationBufferMemory
from agent_tools import get_weather, send_email, web_search
from system_config import GEMINI_API_KEY


def run_smart_agent(user_input: str, conversation_history: list) -> str:
    """
    Runs the smart agent with the given user input and conversation history.

    Args:
        user_input: The user's query string.
        conversation_history: List of dicts with keys "role" and "parts" (each part having "text").
                             e.g., [{"role": "user", "parts": [{"text": "Hello"}]}, ...]

    Returns:
        The agent's response string.
    """
    api_key = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        return "Smart agent is not configured: missing GOOGLE_API_KEY / GEMINI_API_KEY."

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7,
        convert_system_message_to_human=True,
    )

    # STRUCTURED_CHAT supports multi-input tools (send_email has to/subject/body).
    # CHAT_CONVERSATIONAL_REACT_DESCRIPTION does not — that caused the error.
    tools = [get_weather, send_email, web_search]

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    for msg in conversation_history:
        role = msg.get("role")
        parts = msg.get("parts", [])
        if not parts:
            continue
        text = parts[0].get("text", "")
        if not text:
            continue
        if role == "user":
            memory.chat_memory.add_user_message(text)
        elif role == "model":
            memory.chat_memory.add_ai_message(text)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
    )

    try:
        response = agent.invoke({"input": user_input})
        if isinstance(response, dict):
            return response.get("output") or str(response)
        return str(response)
    except Exception as e:
        return f"I encountered an error while processing your request: {e}"
