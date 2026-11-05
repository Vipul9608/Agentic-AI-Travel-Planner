"""TripGenie - Agentic AI Travel Planner (Streamlit front end).

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.agent import AgentResult, build_llm, build_travel_agent, run_travel_agent
from src.config import LLM_PROVIDERS
from src.database import get_database

st.set_page_config(page_title="TripGenie – AI Travel Planner", page_icon="🧳", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def read_secret(name: str) -> str:
    """API key from Streamlit secrets or environment, empty string if absent."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets.toml present
        pass
    return os.getenv(name, "")


@st.cache_resource(show_spinner=False)
def load_database():
    return get_database()


@st.cache_resource(show_spinner=False)
def load_agent(provider: str, model: str, api_key: str, temperature: float):
    """Cache the compiled agent per configuration."""
    return build_travel_agent(build_llm(provider, model, api_key, temperature))


def build_trip_query(form: dict) -> str:
    """Turn the structured form into a natural-language request for the agent."""
    parts = [
        f"Plan a {form['days']}-day trip from {form['origin']} to {form['destination']} "
        f"starting {form['start_date'].isoformat()} for {form['travelers']} traveller(s).",
        f"Flight preference: {form['flight_pref']}.",
        f"Include a return flight to {form['origin']}." if form["return_flight"] else "One-way trip only.",
    ]
    if form["budget"]:
        parts.append(f"Total budget: ₹{form['budget']}.")
    if form["min_stars"] > 1:
        parts.append(f"Hotel should be at least {form['min_stars']}-star.")
    if form["amenities"]:
        parts.append(f"Hotel amenities wanted: {', '.join(form['amenities'])}.")
    if form["interests"]:
        parts.append(f"Interests: {', '.join(form['interests'])}.")
    parts.append(f"Assume ₹{form['daily_expense']} per person per day for food and local travel.")
    if form["notes"].strip():
        parts.append(f"Extra notes: {form['notes'].strip()}")
    return " ".join(parts)


def render_result(result: AgentResult) -> None:
    """Show metrics, the itinerary, structured JSON and the agent's tool trace."""
    data = result.itinerary_json or {}
    budget = data.get("budget") or {}
    hotel = data.get("hotel") or {}
    outbound = (data.get("flights") or {}).get("outbound") or {}
    summary = data.get("trip_summary") or {}

    if data:
        cols = st.columns(4)
        cols[0].metric("Total budget", f"₹{budget.get('total', 0):,}")
        cols[1].metric("Flights", f"₹{budget.get('flights', 0):,}", outbound.get("airline", ""))
        cols[2].metric("Hotel", f"₹{budget.get('hotel', 0):,}", hotel.get("name", ""))
        cols[3].metric("Trip length", f"{summary.get('days', '–')} days", summary.get("destination", ""))

    tab_plan, tab_json, tab_trace = st.tabs(["🗺️ Itinerary", "🧾 Structured JSON", "🧠 Agent reasoning"])
    with tab_plan:
        st.markdown(result.answer_markdown)
        if data.get("budget"):
            breakdown = {k: v for k, v in budget.items() if k != "total" and isinstance(v, (int, float))}
            if breakdown:
                st.caption("Budget split")
                st.bar_chart(pd.DataFrame({"₹": breakdown}))
    with tab_json:
        if data:
            st.json(data)
            st.download_button("Download itinerary JSON", json.dumps(data, indent=2, ensure_ascii=False),
                               file_name="itinerary.json", mime="application/json")
        else:
            st.info("The agent did not return a JSON block for this answer.")
    with tab_trace:
        if not result.tool_calls:
            st.info("No tools were called for this answer.")
        for step, call in enumerate(result.tool_calls, start=1):
            with st.expander(f"Step {step}: `{call.tool_name}`"):
                st.markdown("**Input**")
                st.json(call.arguments)
                st.markdown("**Output**")
                try:
                    st.json(json.loads(call.output))
                except (json.JSONDecodeError, TypeError):
                    st.code(call.output)


def run_and_store(query: str, use_history: bool) -> None:
    """Run the agent, store the result in session state, and surface errors cleanly."""
    if not st.session_state.api_key:
        st.error("Add an API key in the sidebar first.")
        return
    try:
        agent = load_agent(st.session_state.provider, st.session_state.model,
                           st.session_state.api_key, st.session_state.temperature)
        history = st.session_state.get("history") if use_history else None
        with st.spinner("🤖 Agent is searching flights, hotels, places and weather…"):
            result = run_travel_agent(agent, query, history)
        st.session_state.result = result
        st.session_state.history = result.messages
        st.session_state.last_query = query
    except Exception as exc:  # surface provider/network errors to the user
        st.error(f"Agent run failed: {exc}")


# --------------------------------------------------------------------------- #
# Sidebar - model configuration
# --------------------------------------------------------------------------- #
db = load_database()
cities = db.list_cities()

with st.sidebar:
    st.header("⚙️ Model settings")
    provider = st.selectbox("LLM provider", list(LLM_PROVIDERS))
    provider_cfg = LLM_PROVIDERS[provider]
    model = st.text_input("Model", value=provider_cfg["default_model"])
    default_key = read_secret(provider_cfg["key_name"])
    api_key = st.text_input(
        f"{provider_cfg['key_name']}",
        value=default_key,
        type="password",
        help=f"Get a key at {provider_cfg['key_url']}. On Streamlit Cloud, add it under App settings → Secrets.",
    )
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.1)
    st.session_state.update(provider=provider, model=model, api_key=api_key, temperature=temperature)
    if default_key:
        st.success("API key loaded from secrets.")

    st.divider()
    st.markdown(
        "**Agent tools**\n"
        "- ✈️ Flight search (flights.json)\n"
        "- 🏨 Hotel recommender (hotels.json)\n"
        "- 📍 Places discovery (places.json)\n"
        "- 🌤️ Weather (Open-Meteo API)\n"
        "- 💰 Budget estimator"
    )
    if st.button("🔄 Start new conversation", width="stretch"):
        for key in ("result", "history", "last_query"):
            st.session_state.pop(key, None)
        st.rerun()

# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("🧳 TripGenie")
st.caption("Agentic AI travel planner · LangChain tool-calling agent · flights, hotels, places, live weather & budget")

planner_tab, explorer_tab, about_tab = st.tabs(["✨ Plan a trip", "🔎 Data explorer", "ℹ️ How it works"])

with planner_tab:
    mode = st.radio("How do you want to plan?", ["Trip form", "Ask in your own words"], horizontal=True)

    if mode == "Trip form":
        with st.form("trip_form"):
            c1, c2, c3 = st.columns(3)
            origin = c1.selectbox("From", cities, index=cities.index("Bangalore") if "Bangalore" in cities else 0)
            destination = c2.selectbox("To", cities, index=cities.index("Goa") if "Goa" in cities else 1)
            start_date = c3.date_input("Start date", value=date.today() + timedelta(days=7),
                                       min_value=date.today())

            c4, c5, c6 = st.columns(3)
            days = c4.slider("Trip length (days)", 3, 7, 3)
            travelers = c5.number_input("Travellers", 1, 10, 2)
            budget = c6.number_input("Total budget (₹, 0 = no limit)", 0, 1_000_000, 30000, step=1000)

            c7, c8, c9 = st.columns(3)
            flight_pref = c7.selectbox("Flight preference", ["cheapest", "fastest", "balanced"])
            min_stars = c8.slider("Minimum hotel stars", 1, 5, 3)
            daily_expense = c9.number_input("Food & local travel (₹/person/day)", 500, 10000, 1500, step=250)

            c10, c11 = st.columns(2)
            amenities = c10.multiselect("Hotel amenities", db.list_amenities(), default=["wifi"])
            interests = c11.multiselect("Interests", db.list_place_types(), default=["beach", "fort"])

            return_flight = st.checkbox("Include return flight", value=True)
            notes = st.text_area("Anything else?", placeholder="e.g. relaxed pace, prefer morning flights")
            submitted = st.form_submit_button("🚀 Plan my trip", type="primary", width="stretch")

        if submitted:
            if origin == destination:
                st.warning("Origin and destination must be different.")
            else:
                query = build_trip_query({
                    "origin": origin, "destination": destination, "start_date": start_date,
                    "days": days, "travelers": travelers, "budget": budget,
                    "flight_pref": flight_pref, "min_stars": min_stars, "daily_expense": daily_expense,
                    "amenities": amenities, "interests": interests,
                    "return_flight": return_flight, "notes": notes,
                })
                run_and_store(query, use_history=False)
    else:
        examples = [
            "Plan a 4-day trip from Bangalore to Goa next month for 2 people under ₹35,000. We love beaches and museums.",
            "I'm in Mumbai and want a 3-day heritage trip to Jaipur starting 2026-12-20, cheapest options please.",
            "Weekend trip from Delhi to Kolkata for one person, 4-star hotel with pool, budget ₹25k.",
        ]
        example = st.selectbox("Try an example (or write your own below)", [""] + examples)
        free_query = st.text_area("Describe your trip", value=example, height=100)
        if st.button("🚀 Plan my trip", type="primary") and free_query.strip():
            run_and_store(free_query.strip(), use_history=False)

    # ---- Results + follow-up -------------------------------------------- #
    if "result" in st.session_state:
        st.divider()
        with st.expander("Request sent to the agent", expanded=False):
            st.write(st.session_state.last_query)
        render_result(st.session_state.result)

        st.divider()
        st.subheader("💬 Refine this plan")
        with st.form("follow_up", clear_on_submit=True):
            follow_up = st.text_input("Ask a follow-up",
                                      placeholder="e.g. Make it cheaper, or swap Day 2 for more beaches")
            if st.form_submit_button("Send") and follow_up.strip():
                run_and_store(follow_up.strip(), use_history=True)
                st.rerun()

with explorer_tab:
    st.markdown("Browse and filter the datasets the agent uses.")
    dataset = st.radio("Dataset", ["Flights", "Hotels", "Places"], horizontal=True)
    if dataset == "Flights":
        df = pd.DataFrame(db.all_flights())
        f1, f2, f3 = st.columns(3)
        from_sel = f1.multiselect("From", cities)
        to_sel = f2.multiselect("To", cities)
        max_fare = f3.slider("Max fare (₹)", int(df.price.min()), int(df.price.max()), int(df.price.max()))
        if from_sel:
            df = df[df.origin.isin(from_sel)]
        if to_sel:
            df = df[df.destination.isin(to_sel)]
        df = df[df.price <= max_fare]
        st.dataframe(df[["flight_id", "airline", "origin", "destination", "departure_time",
                         "duration", "price"]], width="stretch", hide_index=True)
    elif dataset == "Hotels":
        df = pd.DataFrame(db.all_hotels())
        h1, h2, h3 = st.columns(3)
        city_sel = h1.multiselect("City", cities)
        stars_sel = h2.slider("Min stars", 1, 5, 1)
        max_rate = h3.slider("Max ₹/night", int(df.price_per_night.min()), int(df.price_per_night.max()),
                             int(df.price_per_night.max()))
        if city_sel:
            df = df[df.city.isin(city_sel)]
        df = df[(df.stars >= stars_sel) & (df.price_per_night <= max_rate)]
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        df = pd.DataFrame(db.all_places())
        p1, p2, p3 = st.columns(3)
        city_sel = p1.multiselect("City", cities)
        type_sel = p2.multiselect("Type", db.list_place_types())
        min_rating = p3.slider("Min rating", 0.0, 5.0, 0.0, 0.1)
        if city_sel:
            df = df[df.city.isin(city_sel)]
        if type_sel:
            df = df[df.type.isin(type_sel)]
        df = df[df.rating >= min_rating]
        st.dataframe(df, width="stretch", hide_index=True)
    st.caption(f"{len(df)} rows")

with about_tab:
    st.markdown(
        """
### Problem
Planning a trip means juggling flights, hotels, attractions, weather and budget across many sites.
TripGenie is an **agentic AI** that does that research autonomously and returns one optimised plan.

### Agent workflow (ReAct / tool calling)
1. **Understand** the request (cities, dates, travellers, budget, interests).
2. **Decide** which tools to call and in what order.
3. **Retrieve** data – flights, hotels, places (JSON → SQLite) and live weather (Open-Meteo).
4. **Analyse** – rank cheapest / fastest flights, best-value hotels, top-rated places.
5. **Re-plan** if over budget (cheaper flight / hotel), then **estimate cost**.
6. **Answer** with a day-wise itinerary, weather, budget breakdown, reasons, and JSON.

### Business value
Self-service planning for agencies and aggregators (MakeMyTrip, Booking.com, Ixigo style):
lower support load, personalised recommendations, automated itineraries.
"""
    )
