"""Central configuration: file paths, city coordinates, defaults and LLM providers."""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FLIGHTS_FILE = DATA_DIR / "flights.json"
HOTELS_FILE = DATA_DIR / "hotels.json"
PLACES_FILE = DATA_DIR / "places.json"

# --------------------------------------------------------------------------- #
# Geography - coordinates used for the Open-Meteo weather API
# --------------------------------------------------------------------------- #
CITY_COORDINATES = {
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Delhi": (28.6139, 77.2090),
    "Goa": (15.2993, 74.1240),
    "Hyderabad": (17.3850, 78.4867),
    "Jaipur": (26.9124, 75.7873),
    "Kolkata": (22.5726, 88.3639),
    "Mumbai": (19.0760, 72.8777),
}

# Common alternative spellings users may type.
CITY_ALIASES = {
    "bengaluru": "Bangalore",
    "bombay": "Mumbai",
    "calcutta": "Kolkata",
    "madras": "Chennai",
    "new delhi": "Delhi",
    "panaji": "Goa",
}

# --------------------------------------------------------------------------- #
# Budget defaults (INR)
# --------------------------------------------------------------------------- #
DEFAULT_DAILY_EXPENSE_PER_PERSON = 1500   # food + local transport + entry tickets
GUESTS_PER_ROOM = 2

# --------------------------------------------------------------------------- #
# Weather API
# --------------------------------------------------------------------------- #
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_HORIZON_DAYS = 15          # Open-Meteo forecasts ~16 days ahead
ARCHIVE_LAG_DAYS = 5                # archive data lags a few days behind today
HTTP_TIMEOUT_SECONDS = 15

# WMO weather interpretation codes -> human readable text
WMO_WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

# --------------------------------------------------------------------------- #
# LLM providers (model names are editable in the UI)
# --------------------------------------------------------------------------- #
LLM_PROVIDERS = {
    "Groq (free tier)": {
        "key_name": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "key_url": "https://console.groq.com/keys",
    },
    "OpenAI": {
        "key_name": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "key_url": "https://platform.openai.com/api-keys",
    },
    "Google Gemini": {
        "key_name": "GOOGLE_API_KEY",
        "default_model": "gemini-2.5-flash",
        "key_url": "https://aistudio.google.com/app/apikey",
    },
}
