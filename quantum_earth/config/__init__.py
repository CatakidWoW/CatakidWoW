from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QE_", env_file=".env", extra="ignore")

    app_name: str = "QUANTUM EARTH"
    data_root: Path = Path("data")
    db_path: Path = Path("data/quantum_earth.db")
    default_location: str = "Birmingham"
    default_lat: float = 52.4862
    default_lon: float = -1.8904
    open_meteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    request_timeout_s: float = 30.0
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    reduced_capability: bool = True  # no GPU in current environment


settings = Settings()


KNOWN_LOCATIONS: dict[str, tuple[float, float]] = {
    "birmingham": (52.4862, -1.8904),
    "london": (51.5074, -0.1278),
    "manchester": (53.4808, -2.2426),
    "edinburgh": (55.9533, -3.1883),
    "newyork": (40.7128, -74.0060),
    "tokyo": (35.6762, 139.6503),
    "sydney": (-33.8688, 151.2093),
}


def resolve_location(name_or_coords: str) -> tuple[str, float, float]:
    text = name_or_coords.strip()
    if "," in text:
        lat_s, lon_s = text.split(",", 1)
        return f"{lat_s.strip()},{lon_s.strip()}", float(lat_s), float(lon_s)
    key = text.lower().replace(" ", "")
    if key in KNOWN_LOCATIONS:
        lat, lon = KNOWN_LOCATIONS[key]
        return text.title() if text.islower() else text, lat, lon
    # default Birmingham if unknown name — caller should prefer explicit coords
    return settings.default_location, settings.default_lat, settings.default_lon
