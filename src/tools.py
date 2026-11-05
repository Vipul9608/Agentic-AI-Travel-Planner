"""LangChain tools the travel agent can call.

Each tool returns a JSON string so the LLM receives compact, structured data
and every result includes a short ``reason`` field the agent can use when it
explains "why we selected this".
"""

from __future__ import annotations

import json
import math
from typing import Any, Literal

from langchain_core.tools import tool

from src.config import DEFAULT_DAILY_EXPENSE_PER_PERSON, GUESTS_PER_ROOM
from src.database import get_database
from src.weather import get_weather


def _to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _resolve_city(city: str) -> tuple[str | None, str | None]:
    """Return (canonical_city, error_message)."""
    db = get_database()
    canonical = db.normalise_city(city)
    if canonical is None:
        return None, f"'{city}' is not in the dataset. Available cities: {', '.join(db.list_cities())}."
    return canonical, None


# --------------------------------------------------------------------------- #
@tool
def list_available_cities() -> str:
    """List the cities, hotel amenities and attraction types available in the dataset.
    Call this first if you are unsure whether a city is supported."""
    db = get_database()
    return _to_json({
        "cities": db.list_cities(),
        "hotel_amenities": db.list_amenities(),
        "place_types": db.list_place_types(),
    })


# --------------------------------------------------------------------------- #
@tool
def search_flights(
    origin: str,
    destination: str,
    preference: Literal["cheapest", "fastest", "balanced"] = "cheapest",
    max_price: int | None = None,
) -> str:
    """Search flights between two cities from flights.json.

    Args:
        origin: departure city, e.g. "Delhi".
        destination: arrival city, e.g. "Goa".
        preference: how to rank results - "cheapest", "fastest" or "balanced".
        max_price: optional maximum one-way fare in INR.

    Returns direct flights if any exist; otherwise 1-stop connections.
    Note: dataset dates are sample schedules - use the departure/arrival clock
    times as the daily schedule for the traveller's chosen date.
    """
    db = get_database()
    origin_city, error = _resolve_city(origin)
    if error:
        return _to_json({"status": "error", "message": error})
    destination_city, error = _resolve_city(destination)
    if error:
        return _to_json({"status": "error", "message": error})
    if origin_city == destination_city:
        return _to_json({"status": "error", "message": "Origin and destination are the same city."})

    direct = db.search_flights(origin_city, destination_city, preference, max_price)
    if direct:
        best = direct[0]
        return _to_json({
            "status": "ok",
            "route": f"{origin_city} -> {destination_city}",
            "flight_type": "direct",
            "ranked_by": preference,
            "recommended": best,
            "reason": (f"{best['airline']} {best['flight_id']} is the {preference} direct option "
                       f"(₹{best['price']}, {best['duration']}) out of {len(direct)} direct flight(s)."),
            "alternatives": direct[1:],
        })

    connecting = db.search_connecting_flights(origin_city, destination_city, preference)
    if max_price:
        connecting = [c for c in connecting if c["price"] <= max_price]
    if connecting:
        best = connecting[0]
        return _to_json({
            "status": "ok",
            "route": f"{origin_city} -> {destination_city}",
            "flight_type": "1-stop",
            "ranked_by": preference,
            "recommended": best,
            "reason": (f"No direct flight exists in the dataset, so the {preference} 1-stop route "
                       f"via {best['via']} was chosen (₹{best['price']} total, {best['total_duration']})."),
            "alternatives": connecting[1:],
        })

    return _to_json({
        "status": "not_found",
        "route": f"{origin_city} -> {destination_city}",
        "message": "No direct or 1-stop flights found in the dataset for this route"
                   + (f" under ₹{max_price}." if max_price else "."),
    })


# --------------------------------------------------------------------------- #
@tool
def search_hotels(
    city: str,
    min_stars: int = 0,
    max_price_per_night: int | None = None,
    amenities: list[str] | None = None,
    preference: Literal["best_value", "cheapest", "highest_rated"] = "best_value",
) -> str:
    """Recommend hotels in a city from hotels.json.

    Args:
        city: destination city.
        min_stars: minimum star rating (1-5).
        max_price_per_night: optional nightly budget cap in INR.
        amenities: optional list such as ["wifi", "pool", "breakfast"].
        preference: "best_value" (stars per rupee), "cheapest" or "highest_rated".
    """
    db = get_database()
    canonical, error = _resolve_city(city)
    if error:
        return _to_json({"status": "error", "message": error})

    hotels = db.search_hotels(canonical, min_stars, max_price_per_night, amenities, preference)
    if not hotels:
        # Relax filters so the agent can still offer something.
        relaxed = db.search_hotels(canonical, 0, None, amenities, preference)
        return _to_json({
            "status": "relaxed",
            "message": "No hotel met every filter; showing the closest matches with filters relaxed.",
            "recommended": relaxed[0] if relaxed else None,
            "alternatives": relaxed[1:],
        })

    best = hotels[0]
    reason = (f"{best['name']} ({best['stars']}-star, ₹{best['price_per_night']}/night) ranks first "
              f"by {preference.replace('_', ' ')}")
    if best["matched_amenities"]:
        reason += f" and has {', '.join(best['matched_amenities'])}"
    return _to_json({
        "status": "ok",
        "city": canonical,
        "ranked_by": preference,
        "recommended": best,
        "reason": reason + ".",
        "alternatives": hotels[1:],
    })


# --------------------------------------------------------------------------- #
@tool
def search_places(
    city: str,
    interests: list[str] | None = None,
    min_rating: float = 0.0,
    limit: int = 10,
) -> str:
    """Discover attractions in a city from places.json, ranked by interest match then rating.

    Args:
        city: destination city.
        interests: preferred attraction types, e.g. ["beach", "fort", "museum", "temple"].
        min_rating: minimum rating (0-5).
        limit: maximum number of places to return.
    """
    db = get_database()
    canonical, error = _resolve_city(city)
    if error:
        return _to_json({"status": "error", "message": error})
    places = db.search_places(canonical, interests, min_rating, limit)
    if not places:
        return _to_json({"status": "not_found", "message": f"No attractions in {canonical} match the filters."})
    return _to_json({
        "status": "ok",
        "city": canonical,
        "places": places,
        "note": "Plan 2-3 attractions per day; on arrival/departure days plan 1.",
    })


# --------------------------------------------------------------------------- #
@tool
def get_weather_forecast(city: str, start_date: str, num_days: int) -> str:
    """Get day-wise weather for the trip using the free Open-Meteo API.

    Args:
        city: destination city.
        start_date: first day of the trip, format YYYY-MM-DD.
        num_days: number of trip days (1-14).
    """
    canonical, error = _resolve_city(city)
    if error:
        return _to_json({"status": "error", "message": error})
    return _to_json(get_weather(canonical, start_date, num_days))


# --------------------------------------------------------------------------- #
@tool
def estimate_budget(
    outbound_flight_price: int,
    hotel_price_per_night: int,
    nights: int,
    travelers: int = 1,
    return_flight_price: int = 0,
    daily_expense_per_person: int = DEFAULT_DAILY_EXPENSE_PER_PERSON,
    user_budget: int | None = None,
) -> str:
    """Estimate total trip cost: flights + hotel + per-day local expenses.

    Args:
        outbound_flight_price: one-way fare per person (INR).
        hotel_price_per_night: nightly rate for one room (INR).
        nights: number of hotel nights (trip days - 1).
        travelers: number of people (2 people share a room).
        return_flight_price: return fare per person, 0 if not booked.
        daily_expense_per_person: food + local transport + tickets per day (INR).
        user_budget: optional total budget to compare against.
    """
    travelers = max(1, int(travelers))
    nights = max(0, int(nights))
    days = nights + 1
    rooms = math.ceil(travelers / GUESTS_PER_ROOM)

    flights_total = (int(outbound_flight_price) + int(return_flight_price or 0)) * travelers
    hotel_total = int(hotel_price_per_night) * nights * rooms
    local_total = int(daily_expense_per_person) * days * travelers
    grand_total = flights_total + hotel_total + local_total

    result = {
        "status": "ok",
        "travelers": travelers,
        "rooms": rooms,
        "nights": nights,
        "days": days,
        "breakdown": {
            "flights": flights_total,
            "hotel": hotel_total,
            "food_and_local_travel": local_total,
        },
        "total": grand_total,
        "per_person": round(grand_total / travelers),
        "assumptions": (f"{rooms} room(s) for {travelers} traveller(s); "
                        f"₹{daily_expense_per_person}/person/day for food, local transport and tickets."),
    }
    if user_budget:
        result["user_budget"] = int(user_budget)
        result["within_budget"] = grand_total <= int(user_budget)
        result["difference"] = int(user_budget) - grand_total
    return _to_json(result)


ALL_TOOLS = [
    list_available_cities,
    search_flights,
    search_hotels,
    search_places,
    get_weather_forecast,
    estimate_budget,
]
