# 🧳 TripGenie: Agentic AI Travel Planner (LangChain + Streamlit)

An agentic AI travel assistant. You give it a trip request ("4 days, Bangalore → Goa, 2 people, under ₹35k, love beaches"). The **LangChain tool-calling agent** then works out on its own which tools to call. It searches flights, hotels and attractions, checks the weather, estimates the budget and re-plans if the trip costs too much. The result is a day-wise itinerary in **human-readable markdown** plus **structured JSON**.

## Problem statement
Planning a trip means comparing flights, hotels, attractions, weather and budget across many websites. TripGenie automates that research and gives you one optimised plan that explains its choices.

## Business use case
Travel agencies and aggregators like MakeMyTrip, Booking.com, ClearTrip and Ixigo can use a self-service AI planner to:
- reduce support workload
- personalise recommendations
- design itineraries automatically
- save customers time and money

## Features
| Requirement | Implementation |
|---|---|
| Flight search (JSON) | `search_flights`: filters by route and ranks **cheapest / fastest / balanced**. If no direct flight exists, it falls back to **1-stop** connections |
| Hotel suggestions (JSON) | `search_hotels`: filters by city, stars, price and amenities. Ranks by **best value** (stars per ₹), cheapest or highest-rated |
| Places / POIs (JSON) | `search_places`: ranks by interest match, then rating |
| Live weather | `get_weather_forecast`: free **Open-Meteo** API, no key needed. Uses the live forecast within 16 days, the historical archive for past dates, and a same-dates-last-year estimate further out |
| Budget | `estimate_budget`: flights + hotel (rooms = ⌈travellers/2⌉) + per-day local expenses, compared with the user's budget |
| Agentic reasoning | `langchain.agents.create_agent`, a ReAct-style tool-calling loop |
| Justification | Each tool returns a `reason`. The final answer has a **"Why we selected this"** section |
| Output | Markdown itinerary + JSON (downloadable) + a step-by-step **agent reasoning trace** |
| SQL practices | JSON is loaded into a **normalised, indexed SQLite** database (`src/database.py`) |
| UI | Streamlit with a trip form, free-text mode, follow-up refinement chat and a filterable data explorer |

## Project structure
```
travel-planner-agent/
├── app.py                       # Streamlit UI (entry point)
├── requirements.txt
├── data/
│   ├── flights.json
│   ├── hotels.json
│   └── places.json
├── src/
│   ├── config.py                # paths, city coordinates, providers, defaults
│   ├── database.py              # JSON -> normalised SQLite + search/ranking queries
│   ├── weather.py               # Open-Meteo client (forecast / archive / estimate)
│   ├── tools.py                 # LangChain @tool definitions
│   └── agent.py                 # system prompt, LLM factory, agent build/run/parse
├── tests/
│   └── test_tools.py            # pytest unit tests (no API key needed)
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

## Architecture
```
User ─► Streamlit (app.py) ─► LangChain tool-calling agent (src/agent.py)
                                   │  decides which tools to call, loops until done
        ┌──────────────┬───────────┼─────────────┬──────────────────┐
  search_flights  search_hotels  search_places  get_weather_forecast  estimate_budget
        └──────── SQLite (from JSON) ┘            Open-Meteo API
                                   ▼
            Markdown itinerary + JSON + reasoning trace
```

## Run locally
```bash
git clone https://github.com/<you>/travel-planner-agent.git
cd travel-planner-agent
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your API key
streamlit run app.py
```
You can also paste the API key into the sidebar while the app is running.

### Which LLM key?
You only need **one**. Pick the provider in the sidebar:

| Provider | Secret name | Default model | Get a key |
|---|---|---|---|
| Groq (free tier) | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | https://console.groq.com/keys |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` | https://platform.openai.com/api-keys |
| Google Gemini | `GOOGLE_API_KEY` | `gemini-2.5-flash` | https://aistudio.google.com/app/apikey |

The model name can be edited in the sidebar if a provider retires a model.

## Deploy on Streamlit Community Cloud
1. Push this folder to a **public GitHub repo**. `.gitignore` already keeps `secrets.toml` out of git.
2. Go to https://share.streamlit.io → **Create app** → choose your repo and branch. Set **Main file path** to `app.py`.
3. Under **Advanced settings**, choose Python **3.11 or 3.12**. In **Secrets**, paste for example:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
4. Click **Deploy**. The sidebar will show "API key loaded from secrets".

## Tests
```bash
pip install pytest
python -m pytest -q
```

## Example output (illustrative — weather varies by date)
```
# 🧳 Your 3-Day Trip to Goa (2026-11-12 – 2026-11-14)
✈️ Flight: Go First FL0021, Bangalore 07:27 → Goa 09:27 (2h) – ₹5,356
🏨 Hotel: Royal Heritage, 5-star, ₹1,232/night (wifi, pool, breakfast, parking)
🌤️ Weather: Day 1 Partly cloudy 32°C · Day 2 Clear sky 33°C · Day 3 Slight rain 31°C
🗓️ Day 1: Beautiful Park (fort, 4.3) · Day 2: Famous Park (museum, 4.5), Popular Lake (4.2) …
💰 Flights ₹10,712 · Hotel ₹2,464 · Food & local ₹9,000 · Total ₹22,176
```

## Notes and limitations
- Dates in the sample flight dataset are random. The agent treats each flight's clock times as a daily schedule on the traveller's chosen date.
- The dataset is small: some routes, such as Delhi → Goa, have no direct or 1-stop flight. The agent says so clearly and still plans the stay.
- Extend coverage by adding records to `data/*.json` and coordinates for new cities in `src/config.py`.
