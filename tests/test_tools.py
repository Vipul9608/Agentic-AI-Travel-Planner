"""Unit tests for the data layer and tools (no LLM or network needed)."""

import json
from unittest.mock import patch

from src.agent import split_answer
from src.database import get_database
from src.tools import estimate_budget, search_flights, search_hotels, search_places


def test_database_loads_all_records():
    db = get_database()
    assert len(db.all_flights()) == 30
    assert len(db.all_hotels()) == 40
    assert len(db.all_places()) == 40
    assert "Goa" in db.list_cities()


def test_city_aliases():
    assert get_database().normalise_city(" bengaluru ") == "Bangalore"
    assert get_database().normalise_city("Atlantis") is None


def test_direct_flight_cheapest():
    result = json.loads(search_flights.invoke({"origin": "Delhi", "destination": "Kolkata"}))
    assert result["flight_type"] == "direct"
    assert result["recommended"]["price"] == 3779


def test_connecting_flight_when_no_direct():
    result = json.loads(search_flights.invoke({"origin": "Delhi", "destination": "Goa"}))
    assert result["status"] in {"ok", "not_found"}
    if result["status"] == "ok":
        assert result["flight_type"] == "1-stop"


def test_hotel_filters():
    result = json.loads(search_hotels.invoke({"city": "Goa", "min_stars": 4, "amenities": ["wifi"]}))
    hotel = result["recommended"]
    assert hotel["stars"] >= 4 and "wifi" in hotel["amenities"]


def test_places_interest_first():
    result = json.loads(search_places.invoke({"city": "Goa", "interests": ["fort"]}))
    assert result["places"][0]["type"] == "fort"


def test_budget_math():
    result = json.loads(estimate_budget.invoke({
        "outbound_flight_price": 5000, "return_flight_price": 4000,
        "hotel_price_per_night": 3000, "nights": 2, "travelers": 2,
        "daily_expense_per_person": 1000, "user_budget": 40000,
    }))
    assert result["breakdown"] == {"flights": 18000, "hotel": 6000, "food_and_local_travel": 6000}
    assert result["total"] == 30000 and result["within_budget"]


def test_weather_failure_is_graceful():
    import requests
    from src.weather import get_weather
    with patch("src.weather.requests.get", side_effect=requests.ConnectionError("offline")):
        result = get_weather("Goa", "2030-01-01", 3)
    assert result["status"] == "unavailable"


def test_split_answer_extracts_json():
    md, data = split_answer('# Trip\nhello\n```json\n{"budget": {"total": 1}}\n```')
    assert md == "# Trip\nhello" and data["budget"]["total"] == 1
