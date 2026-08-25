# QUANTUM EARTH

Autonomous Earth-System Prediction Intelligence Platform.

**Objective:** Maintain the most accurate continuously updated probabilistic representation of the future state of the observable Earth that can be produced from available data, models, computation, and scientific knowledge.

The system never claims certainty where uncertainty exists.

## Current status (Stage 1 + Vertical Slice A)

Implemented and real:

- Repository architecture, configuration, provenance, scientific integrity enums
- Data-source registry with licence / coverage / reliability metadata
- Observation + EarthState schemas (estimate, uncertainty, provenance, quality)
- Model / experiment registries with promotion statuses
- Verification framework (MAE, RMSE, bias, CRPS-like ensemble score)
- System health + Quantum Assurance (GREEN / YELLOW / RED)
- **Real** Open-Meteo forecast + archive connectors (free, no fabricated API)
- Earth-state estimation from observations
- Baseline models: persistence, climatology, AR(1)
- Multi-model ensemble forecasts (Open-Meteo ECMWF / GFS / ICON members)
- Forecast API, CLI, and Earth Intelligence dashboard
- Aggressive unit + adversarial tests

Explicitly **not** claimed as operational yet: satellite adapters, radar, hydrology, phenology, fire, ocean, space weather, geohazards, autonomous research loop. See `docs/REAL_VS_SIMULATED.md`.

## Quick start

```bash
pip install -e ".[dev]"
quantum-earth status
quantum-earth forecast Birmingham --hours 48
quantum-earth verify --location Birmingham --hours 24
uvicorn quantum_earth.api.app:app --host 0.0.0.0 --port 8080
```

Dashboard: http://localhost:8080/

## Hardware note

This environment runs on CPU only (no NVIDIA GPU). Reduced-capability mode is the default.
