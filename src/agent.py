"""Tool-calling travel agent built with LangChain (``langchain.agents.create_agent``).

The agent follows a ReAct-style loop: the LLM reasons about the request,
decides which tool to call, reads the result, and repeats until it can write
the final itinerary (markdown for humans + a JSON block for machines).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from src.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

RECURSION_LIMIT = 30  # max graph steps (each tool call = 2 steps)

SYSTEM_PROMPT = """You are TripGenie, an expert Indian travel planner agent.
Today's date is {today}. All prices are in Indian Rupees (₹).

## How you work
1. Understand the request: origin, destination, start date, trip length (3-7 days,
   default 3), travellers (default 1), budget, interests, hotel needs.
   If the start date is missing, assume a date 7 days from today and say so.
2. ALWAYS gather facts with tools - never invent flights, hotels, places or weather:
   - search_flights for origin -> destination (and destination -> origin for the return).
   - search_hotels in the destination (respect stars / budget / amenities).
   - search_places in the destination (use the traveller's interests).
   - get_weather_forecast for the trip dates.
   - estimate_budget with the chosen flight + hotel prices (nights = days - 1).
   If a city is unknown, call list_available_cities and tell the user what is supported.
3. If the total exceeds the user's budget, try cheaper options (cheapest flight,
   cheaper hotel) and re-run estimate_budget before answering.
4. If no flight exists for a leg, say so clearly, still plan the stay, and leave
   that flight out of the budget (use null in the JSON).
5. Dataset flight dates are only sample schedules: present the flight on the
   traveller's travel date using the clock times from the data.

## Final answer format (markdown, in this order)
# 🧳 Your N-Day Trip to <City> (<dates>)
### 📋 Trip Summary  (2-3 lines)
### ✈️ Flight Selected  (outbound + return: airline, flight id, times, duration, price)
### 🏨 Hotel Recommendation  (name, stars, ₹/night, amenities)
### 🌤️ Weather  (one line per day, mention the data source)
### 🗓️ Day-wise Itinerary  (Day 1..N, 1-3 attractions/day with ratings; arrival & departure days lighter)
### 💰 Budget Breakdown  (flights, hotel, food & local travel, **Total**, budget status)
### 🤔 Why We Selected This  (short bullets justifying flight, hotel and places with numbers)

Then end with ONE fenced ```json block with exactly these keys:
{{"trip_summary": {{"origin": str, "destination": str, "start_date": "YYYY-MM-DD",
   "end_date": "YYYY-MM-DD", "days": int, "travelers": int}},
 "flights": {{"outbound": {{"airline": str, "flight_id": str, "departure": str, "arrival": str, "price": int}},
             "return": {{...same keys...}} or null}},
 "hotel": {{"name": str, "hotel_id": str, "stars": int, "price_per_night": int, "amenities": [str]}},
 "weather": [{{"date": "YYYY-MM-DD", "condition": str, "temp_max_c": float, "temp_min_c": float}}],
 "itinerary": [{{"day": int, "date": "YYYY-MM-DD", "activities": [str]}}],
 "budget": {{"flights": int, "hotel": int, "food_and_local_travel": int, "total": int}},
 "reasoning": [str]}}

For general travel questions that are not trip-planning requests, answer briefly
and skip the JSON block.
"""


@dataclass
class ToolCallRecord:
    """One step of the agent's reasoning trace."""
    tool_name: str
    arguments: dict[str, Any]
    output: str


@dataclass
class AgentResult:
    """Everything the UI needs from one agent run."""
    answer_markdown: str
    itinerary_json: dict[str, Any] | None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    messages: list[BaseMessage] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# LLM factory
# --------------------------------------------------------------------------- #
def build_llm(provider: str, model: str, api_key: str, temperature: float = 0.2) -> BaseChatModel:
    """Create a tool-calling chat model for the chosen provider."""
    if not api_key:
        raise ValueError(f"An API key is required for {provider}.")
    if provider.startswith("Groq"):
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, api_key=api_key, temperature=temperature)
    if provider.startswith("OpenAI"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature)
    if provider.startswith("Google"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=temperature)
    raise ValueError(f"Unsupported provider: {provider}")


def build_travel_agent(llm: BaseChatModel):
    """Compile the LangChain tool-calling agent graph."""
    prompt = SYSTEM_PROMPT.format(today=date.today().isoformat())
    return create_agent(model=llm, tools=ALL_TOOLS, system_prompt=prompt)


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #
_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def _message_text(message: BaseMessage) -> str:
    """Normalise message content (some providers return a list of parts)."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "".join(parts)


def split_answer(text: str) -> tuple[str, dict[str, Any] | None]:
    """Separate the human-readable markdown from the trailing JSON block."""
    matches = list(_JSON_BLOCK.finditer(text))
    if not matches:
        return text.strip(), None
    last = matches[-1]
    try:
        data = json.loads(last.group(1))
    except json.JSONDecodeError:
        logger.warning("Agent returned malformed JSON block")
        return text.strip(), None
    markdown = (text[: last.start()] + text[last.end():]).strip()
    return markdown, data


def extract_tool_calls(messages: list[BaseMessage]) -> list[ToolCallRecord]:
    """Pair each AI tool call with the tool's output for the reasoning trace."""
    outputs = {m.tool_call_id: _message_text(m) for m in messages if isinstance(m, ToolMessage)}
    records = []
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                records.append(ToolCallRecord(call["name"], call.get("args", {}), outputs.get(call["id"], "")))
    return records


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run_travel_agent(agent, user_query: str, history: list[BaseMessage] | None = None) -> AgentResult:
    """Invoke the agent on a query (with optional chat history) and parse the result."""
    input_messages = list(history or []) + [HumanMessage(content=user_query)]
    state = agent.invoke({"messages": input_messages}, config={"recursion_limit": RECURSION_LIMIT})
    messages: list[BaseMessage] = state["messages"]
    new_messages = messages[len(input_messages):]

    final_text = ""
    for message in reversed(new_messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            final_text = _message_text(message)
            break

    markdown, itinerary = split_answer(final_text or "The agent did not return an answer.")
    return AgentResult(
        answer_markdown=markdown,
        itinerary_json=itinerary,
        tool_calls=extract_tool_calls(new_messages),
        messages=messages,
    )
