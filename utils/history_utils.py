"""Conversion helpers between Streamlit's chat history and LangChain messages."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

ChatTurn = tuple[str, str]  # (role, content) where role is "user" or "assistant"


def to_langchain_messages(history: list[ChatTurn], max_turns: int = 6) -> list[BaseMessage]:
    """Convert the last `max_turns` chat turns into LangChain messages.

    Limiting to recent turns keeps the prompt compact while still giving
    the model enough context to resolve follow-up questions.
    """
    recent = history[-max_turns:] if max_turns else history
    messages: list[BaseMessage] = []
    for role, content in recent:
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages
