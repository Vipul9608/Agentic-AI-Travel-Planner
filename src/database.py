"""SQLite data layer.

The three JSON datasets are loaded into a normalised, indexed, in-memory SQLite
database. All agent tools query the data through the ``TravelDatabase`` class,
so filtering / ranking logic lives in one place and is easy to test.

Schema (normalised):
    cities(city_id, city_name)
    airlines(airline_id, airline_name)
    flights(flight_id, airline_id, origin_city_id, destination_city_id,
            departure_time, arrival_time, duration_minutes, price)
    hotels(hotel_id, hotel_name, city_id, stars, price_per_night)
    amenities(amenity_id, amenity_name)
    hotel_amenities(hotel_id, amenity_id)
    places(place_id, place_name, city_id, place_type, rating)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import CITY_ALIASES, FLIGHTS_FILE, HOTELS_FILE, PLACES_FILE

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE cities (
    city_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    city_name    TEXT NOT NULL UNIQUE COLLATE NOCASE
);
CREATE TABLE airlines (
    airline_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    airline_name TEXT NOT NULL UNIQUE
);
CREATE TABLE flights (
    flight_id           TEXT PRIMARY KEY,
    airline_id          INTEGER NOT NULL REFERENCES airlines(airline_id),
    origin_city_id      INTEGER NOT NULL REFERENCES cities(city_id),
    destination_city_id INTEGER NOT NULL REFERENCES cities(city_id),
    departure_time      TEXT NOT NULL,
    arrival_time        TEXT NOT NULL,
    duration_minutes    INTEGER NOT NULL,
    price               INTEGER NOT NULL
);
CREATE TABLE hotels (
    hotel_id        TEXT PRIMARY KEY,
    hotel_name      TEXT NOT NULL,
    city_id         INTEGER NOT NULL REFERENCES cities(city_id),
    stars           INTEGER NOT NULL,
    price_per_night INTEGER NOT NULL
);
CREATE TABLE amenities (
    amenity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    amenity_name TEXT NOT NULL UNIQUE
);
CREATE TABLE hotel_amenities (
    hotel_id   TEXT    NOT NULL REFERENCES hotels(hotel_id),
    amenity_id INTEGER NOT NULL REFERENCES amenities(amenity_id),
    PRIMARY KEY (hotel_id, amenity_id)
);
CREATE TABLE places (
    place_id   TEXT PRIMARY KEY,
    place_name TEXT NOT NULL,
    city_id    INTEGER NOT NULL REFERENCES cities(city_id),
    place_type TEXT NOT NULL,
    rating     REAL NOT NULL
);

CREATE INDEX idx_flights_route   ON flights(origin_city_id, destination_city_id);
CREATE INDEX idx_flights_price   ON flights(price);
CREATE INDEX idx_hotels_city     ON hotels(city_id, price_per_night);
CREATE INDEX idx_places_city     ON places(city_id, rating DESC);
CREATE INDEX idx_places_type     ON places(place_type);
"""

FLIGHT_SELECT_SQL = """
SELECT f.flight_id, a.airline_name AS airline, o.city_name AS origin,
       d.city_name AS destination, f.departure_time, f.arrival_time,
       f.duration_minutes, f.price
FROM flights f
JOIN airlines a ON a.airline_id = f.airline_id
JOIN cities   o ON o.city_id    = f.origin_city_id
JOIN cities   d ON d.city_id    = f.destination_city_id
"""


def _load_json(path: Path) -> list[dict[str, Any]]:
    """Read a JSON list from disk with a helpful error message."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset is not valid JSON: {path} ({exc})") from exc


def _minutes_between(start_iso: str, end_iso: str) -> int:
    """Flight duration in minutes from two ISO timestamps."""
    start = datetime.fromisoformat(start_iso)
    end = datetime.fromisoformat(end_iso)
    return int((end - start).total_seconds() // 60)


def _format_duration(minutes: int) -> str:
    hours, mins = divmod(int(minutes), 60)
    return f"{hours}h {mins:02d}m"


class TravelDatabase:
    """Normalised SQLite store built from flights/hotels/places JSON files."""

    def __init__(
        self,
        flights_file: Path = FLIGHTS_FILE,
        hotels_file: Path = HOTELS_FILE,
        places_file: Path = PLACES_FILE,
    ) -> None:
        # check_same_thread=False: Streamlit may call from different threads.
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        try:
            self._conn.executescript(SCHEMA_SQL)
            self._load_flights(_load_json(flights_file))
            self._load_hotels(_load_json(hotels_file))
            self._load_places(_load_json(places_file))
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.exception("Failed to build the travel database")
            raise RuntimeError(f"Database initialisation failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Loading helpers
    # ------------------------------------------------------------------ #
    def _get_or_create(self, table: str, id_col: str, name_col: str, value: str) -> int:
        row = self._conn.execute(
            f"SELECT {id_col} FROM {table} WHERE {name_col} = ?", (value,)
        ).fetchone()
        if row:
            return row[0]
        cursor = self._conn.execute(f"INSERT INTO {table} ({name_col}) VALUES (?)", (value,))
        return cursor.lastrowid

    def _city_id(self, city: str) -> int:
        return self._get_or_create("cities", "city_id", "city_name", city.strip())

    def _load_flights(self, flights: list[dict[str, Any]]) -> None:
        for flight in flights:
            self._conn.execute(
                "INSERT INTO flights VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    flight["flight_id"],
                    self._get_or_create("airlines", "airline_id", "airline_name", flight["airline"]),
                    self._city_id(flight["from"]),
                    self._city_id(flight["to"]),
                    flight["departure_time"],
                    flight["arrival_time"],
                    _minutes_between(flight["departure_time"], flight["arrival_time"]),
                    int(flight["price"]),
                ),
            )

    def _load_hotels(self, hotels: list[dict[str, Any]]) -> None:
        for hotel in hotels:
            self._conn.execute(
                "INSERT INTO hotels VALUES (?, ?, ?, ?, ?)",
                (
                    hotel["hotel_id"],
                    hotel["name"],
                    self._city_id(hotel["city"]),
                    int(hotel["stars"]),
                    int(hotel["price_per_night"]),
                ),
            )
            for amenity in hotel.get("amenities", []):
                amenity_id = self._get_or_create(
                    "amenities", "amenity_id", "amenity_name", amenity.lower()
                )
                self._conn.execute(
                    "INSERT OR IGNORE INTO hotel_amenities VALUES (?, ?)",
                    (hotel["hotel_id"], amenity_id),
                )

    def _load_places(self, places: list[dict[str, Any]]) -> None:
        for place in places:
            self._conn.execute(
                "INSERT INTO places VALUES (?, ?, ?, ?, ?)",
                (
                    place["place_id"],
                    place["name"],
                    self._city_id(place["city"]),
                    place["type"].lower(),
                    float(place["rating"]),
                ),
            )

    def _query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        try:
            with self._lock:
                return [dict(row) for row in self._conn.execute(sql, params).fetchall()]
        except sqlite3.Error as exc:
            logger.exception("SQL query failed: %s", sql)
            raise RuntimeError(f"Database query failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Lookups
    # ------------------------------------------------------------------ #
    def list_cities(self) -> list[str]:
        return [row["city_name"] for row in self._query("SELECT city_name FROM cities ORDER BY city_name")]

    def list_amenities(self) -> list[str]:
        rows = self._query("SELECT amenity_name FROM amenities ORDER BY amenity_name")
        return [row["amenity_name"] for row in rows]

    def list_place_types(self) -> list[str]:
        rows = self._query("SELECT DISTINCT place_type FROM places ORDER BY place_type")
        return [row["place_type"] for row in rows]

    def normalise_city(self, city: str) -> str | None:
        """Map user input ('bengaluru', ' goa ') to a canonical dataset city."""
        if not city:
            return None
        cleaned = city.strip().lower()
        cleaned = CITY_ALIASES.get(cleaned, cleaned)
        for known in self.list_cities():
            if known.lower() == str(cleaned).lower():
                return known
        return None

    # ------------------------------------------------------------------ #
    # Flights
    # ------------------------------------------------------------------ #
    def search_flights(
        self,
        origin: str,
        destination: str,
        sort_by: str = "cheapest",
        max_price: int | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Direct flights on a route, ranked by price, duration or a balance of both."""
        sql = FLIGHT_SELECT_SQL + " WHERE o.city_name = ? AND d.city_name = ?"
        params: list[Any] = [origin, destination]
        if max_price:
            sql += " AND f.price <= ?"
            params.append(int(max_price))
        flights = self._query(sql, tuple(params))
        flights = self._rank_flights(flights, sort_by)
        return [self._decorate_flight(f) for f in flights[:limit]]

    def search_connecting_flights(
        self, origin: str, destination: str, sort_by: str = "cheapest", limit: int = 3
    ) -> list[dict[str, Any]]:
        """One-stop itineraries (origin -> hub -> destination).

        Dataset dates are random samples, so the departure *time of day* is
        treated as a daily schedule. Layover = time from leg-1 arrival to the
        next leg-2 departure (at least 60 minutes, otherwise next day).
        """
        first_legs = self._query(FLIGHT_SELECT_SQL + " WHERE o.city_name = ? AND d.city_name != ?",
                                 (origin, destination))
        options = []
        for leg1 in first_legs:
            second_legs = self._query(FLIGHT_SELECT_SQL + " WHERE o.city_name = ? AND d.city_name = ?",
                                      (leg1["destination"], destination))
            for leg2 in second_legs:
                arrival = datetime.fromisoformat(leg1["arrival_time"])
                dep2 = datetime.fromisoformat(leg2["departure_time"])
                next_dep = arrival.replace(hour=dep2.hour, minute=dep2.minute, second=0)
                if next_dep - arrival < timedelta(minutes=60):
                    next_dep += timedelta(days=1)
                layover = int((next_dep - arrival).total_seconds() // 60)
                total = leg1["duration_minutes"] + layover + leg2["duration_minutes"]
                options.append({
                    "type": "1-stop",
                    "via": leg1["destination"],
                    "legs": [self._decorate_flight(leg1), self._decorate_flight(leg2)],
                    "price": leg1["price"] + leg2["price"],
                    "duration_minutes": total,
                    "total_duration": _format_duration(total),
                    "layover": _format_duration(layover),
                })
        options = self._rank_flights(options, sort_by)
        return options[:limit]

    @staticmethod
    def _rank_flights(flights: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
        if not flights:
            return flights
        sort_by = (sort_by or "cheapest").lower()
        if sort_by == "fastest":
            return sorted(flights, key=lambda f: (f["duration_minutes"], f["price"]))
        if sort_by == "balanced":
            min_price = min(f["price"] for f in flights)
            min_time = min(f["duration_minutes"] for f in flights)
            return sorted(
                flights,
                key=lambda f: 0.6 * f["price"] / min_price + 0.4 * f["duration_minutes"] / min_time,
            )
        return sorted(flights, key=lambda f: (f["price"], f["duration_minutes"]))

    @staticmethod
    def _decorate_flight(flight: dict[str, Any]) -> dict[str, Any]:
        departure = datetime.fromisoformat(flight["departure_time"])
        arrival = datetime.fromisoformat(flight["arrival_time"])
        return {
            **flight,
            "departure_clock": departure.strftime("%H:%M"),
            "arrival_clock": arrival.strftime("%H:%M") + (" (+1 day)" if arrival.date() > departure.date() else ""),
            "duration": _format_duration(flight["duration_minutes"]),
        }

    # ------------------------------------------------------------------ #
    # Hotels
    # ------------------------------------------------------------------ #
    def search_hotels(
        self,
        city: str,
        min_stars: int = 0,
        max_price_per_night: int | None = None,
        required_amenities: list[str] | None = None,
        sort_by: str = "best_value",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Hotels in a city filtered by stars/price/amenities and ranked."""
        sql = """
            SELECT h.hotel_id, h.hotel_name AS name, c.city_name AS city, h.stars,
                   h.price_per_night,
                   GROUP_CONCAT(a.amenity_name, ', ') AS amenities
            FROM hotels h
            JOIN cities c ON c.city_id = h.city_id
            LEFT JOIN hotel_amenities ha ON ha.hotel_id = h.hotel_id
            LEFT JOIN amenities a ON a.amenity_id = ha.amenity_id
            WHERE c.city_name = ? AND h.stars >= ?
        """
        params: list[Any] = [city, int(min_stars or 0)]
        if max_price_per_night:
            sql += " AND h.price_per_night <= ?"
            params.append(int(max_price_per_night))
        sql += " GROUP BY h.hotel_id"
        hotels = self._query(sql, tuple(params))

        wanted = {a.strip().lower() for a in (required_amenities or []) if a.strip()}
        for hotel in hotels:
            hotel_amenities = {a.strip() for a in (hotel["amenities"] or "").split(",") if a.strip()}
            hotel["amenities"] = sorted(hotel_amenities)
            hotel["matched_amenities"] = sorted(wanted & hotel_amenities)
            hotel["missing_amenities"] = sorted(wanted - hotel_amenities)
            # Value score: stars delivered per ₹1,000 spent, plus amenity richness.
            hotel["value_score"] = round(
                hotel["stars"] / (hotel["price_per_night"] / 1000) + 0.1 * len(hotel_amenities), 2
            )
        if wanted:
            # Prefer hotels that satisfy all requested amenities.
            hotels = [h for h in hotels if not h["missing_amenities"]] or hotels

        sort_by = (sort_by or "best_value").lower()
        if sort_by == "cheapest":
            hotels.sort(key=lambda h: h["price_per_night"])
        elif sort_by in {"highest_rated", "stars"}:
            hotels.sort(key=lambda h: (-h["stars"], h["price_per_night"]))
        else:
            hotels.sort(key=lambda h: (len(h["missing_amenities"]), -h["value_score"]))
        return hotels[:limit]

    # ------------------------------------------------------------------ #
    # Places
    # ------------------------------------------------------------------ #
    def search_places(
        self,
        city: str,
        place_types: list[str] | None = None,
        min_rating: float = 0.0,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Attractions in a city, preferred types first, then by rating."""
        rows = self._query(
            """
            SELECT p.place_id, p.place_name AS name, c.city_name AS city,
                   p.place_type AS type, p.rating
            FROM places p JOIN cities c ON c.city_id = p.city_id
            WHERE c.city_name = ? AND p.rating >= ?
            ORDER BY p.rating DESC
            """,
            (city, float(min_rating or 0)),
        )
        preferred = {t.strip().lower() for t in (place_types or []) if t.strip()}
        for row in rows:
            row["matches_interest"] = row["type"] in preferred
        rows.sort(key=lambda r: (not r["matches_interest"], -r["rating"]))
        return rows[:limit]

    # ------------------------------------------------------------------ #
    # Raw tables for the data explorer
    # ------------------------------------------------------------------ #
    def all_flights(self) -> list[dict[str, Any]]:
        return [self._decorate_flight(f) for f in self._query(FLIGHT_SELECT_SQL + " ORDER BY f.flight_id")]

    def all_hotels(self) -> list[dict[str, Any]]:
        return self._query(
            """
            SELECT h.hotel_id, h.hotel_name AS name, c.city_name AS city, h.stars,
                   h.price_per_night, GROUP_CONCAT(a.amenity_name, ', ') AS amenities
            FROM hotels h JOIN cities c ON c.city_id = h.city_id
            LEFT JOIN hotel_amenities ha ON ha.hotel_id = h.hotel_id
            LEFT JOIN amenities a ON a.amenity_id = ha.amenity_id
            GROUP BY h.hotel_id ORDER BY h.hotel_id
            """
        )

    def all_places(self) -> list[dict[str, Any]]:
        return self._query(
            """
            SELECT p.place_id, p.place_name AS name, c.city_name AS city,
                   p.place_type AS type, p.rating
            FROM places p JOIN cities c ON c.city_id = p.city_id ORDER BY p.place_id
            """
        )


@lru_cache(maxsize=1)
def get_database() -> TravelDatabase:
    """Process-wide singleton so the JSON files are parsed only once."""
    return TravelDatabase()
