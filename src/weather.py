"""Weather lookups using the free Open-Meteo API (no API key needed).

Strategy by travel date:
* within the next ~16 days  -> live forecast   (api.open-meteo.com/v1/forecast)
* in the past               -> observed data   (archive-api.open-meteo.com/v1/archive)
* further in the future     -> same dates last year from the archive,
                               returned as a "historical estimate"
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import requests

from src.config import (
    ARCHIVE_LAG_DAYS,
    CITY_COORDINATES,
    FORECAST_HORIZON_DAYS,
    HTTP_TIMEOUT_SECONDS,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_FORECAST_URL,
    WMO_WEATHER_CODES,
)

logger = logging.getLogger(__name__)


def _shift_year(day: date, years: int) -> date:
    """Shift a date by whole years, handling 29 February."""
    try:
        return day.replace(year=day.year + years)
    except ValueError:
        return day.replace(year=day.year + years, day=28)


def _call_open_meteo(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise ValueError(payload.get("reason", "Open-Meteo returned an error"))
    return payload


def _parse_daily(payload: dict[str, Any], display_dates: list[date]) -> list[dict[str, Any]]:
    daily = payload.get("daily", {})
    codes = daily.get("weather_code") or daily.get("weathercode") or []
    t_max = daily.get("temperature_2m_max", [])
    t_min = daily.get("temperature_2m_min", [])
    rain = daily.get("precipitation_sum", [])
    rain_prob = daily.get("precipitation_probability_max", [])

    days = []
    for i, display_date in enumerate(display_dates):
        if i >= len(t_max):
            break
        code = codes[i] if i < len(codes) else None
        day = {
            "date": display_date.isoformat(),
            "condition": WMO_WEATHER_CODES.get(code, "Unknown") if code is not None else "Unknown",
            "temp_max_c": t_max[i],
            "temp_min_c": t_min[i] if i < len(t_min) else None,
            "precipitation_mm": rain[i] if i < len(rain) else None,
        }
        if i < len(rain_prob) and rain_prob[i] is not None:
            day["rain_probability_pct"] = rain_prob[i]
        days.append(day)
    return days


def get_weather(city: str, start_date: str, num_days: int) -> dict[str, Any]:
    """Return day-wise weather for ``num_days`` starting at ``start_date`` (YYYY-MM-DD)."""
    if city not in CITY_COORDINATES:
        return {"status": "error", "message": f"No coordinates configured for '{city}'."}
    try:
        start = date.fromisoformat(start_date)
    except ValueError:
        return {"status": "error", "message": f"Invalid date '{start_date}'. Use YYYY-MM-DD."}

    num_days = max(1, min(int(num_days), 14))
    end = start + timedelta(days=num_days - 1)
    display_dates = [start + timedelta(days=i) for i in range(num_days)]
    latitude, longitude = CITY_COORDINATES[city]
    today = date.today()

    base_params = {"latitude": latitude, "longitude": longitude, "timezone": "auto"}
    daily_fields = "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum"

    try:
        if today - timedelta(days=ARCHIVE_LAG_DAYS) <= start and end <= today + timedelta(days=FORECAST_HORIZON_DAYS):
            source = "Open-Meteo live forecast"
            payload = _call_open_meteo(OPEN_METEO_FORECAST_URL, {
                **base_params,
                "daily": daily_fields + ",precipitation_probability_max",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            })
        elif end < today - timedelta(days=ARCHIVE_LAG_DAYS):
            source = "Open-Meteo historical observations"
            payload = _call_open_meteo(OPEN_METEO_ARCHIVE_URL, {
                **base_params, "daily": daily_fields,
                "start_date": start.isoformat(), "end_date": end.isoformat(),
            })
        else:
            # Beyond the forecast horizon: use the same dates in the most recent
            # past year as a climate-based estimate.
            years_back = 1
            while _shift_year(end, -years_back) >= today - timedelta(days=ARCHIVE_LAG_DAYS):
                years_back += 1
            ref_start, ref_end = _shift_year(start, -years_back), _shift_year(end, -years_back)
            source = (f"Historical estimate (actual weather on the same dates in "
                      f"{ref_start.year}; live forecast available ~16 days before travel)")
            payload = _call_open_meteo(OPEN_METEO_ARCHIVE_URL, {
                **base_params, "daily": daily_fields,
                "start_date": ref_start.isoformat(), "end_date": ref_end.isoformat(),
            })
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Weather lookup failed for %s: %s", city, exc)
        return {
            "status": "unavailable",
            "city": city,
            "message": f"Weather service could not be reached ({exc.__class__.__name__}). "
                       "Advise the traveller to check a forecast closer to the date.",
        }

    return {
        "status": "ok",
        "city": city,
        "source": source,
        "days": _parse_daily(payload, display_dates),
    }
