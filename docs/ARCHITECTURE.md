# QUANTUM EARTH Architecture (Stage 1)

## Design philosophy

Ecosystem of specialised modules connected through a common `EarthState` representation.
Not one enormous neural network. Not an LLM chatbot.

Scientific integrity categories are first-class:

| Category | Meaning |
|----------|---------|
| OBSERVED | Direct measurement / ingested observation product |
| INFERRED | Derived by assimilation / estimation |
| FORECAST | Predictive output |
| SCENARIO | Counterfactual / what-if |
| HYPOTHESIS | Unvalidated research claim |
| UNKNOWN | Insufficient evidence |

## Package layout

```
quantum_earth/
  core/           schemas, provenance, integrity, geometry
  config/         settings
  data/           connectors, registry, ingestion, validation
  state/          EarthState engine
  assimilation/   assimilation interfaces (stubs + simple blend)
  models/         registry + baseline families
  forecast/       horizons, ensembles, orchestration
  verification/   metrics + rolling evaluation
  assurance/      Quantum Assurance GREEN/YELLOW/RED
  health/         heartbeats, fallbacks
  storage/        RAW/PROCESSED/FORECAST/... artefact layout
  orchestration/  autonomous operating loop (Stage 1 skeleton)
  api/            FastAPI surface
  cli/            Typer CLI
  dashboard/      Earth intelligence command centre UI
```

## First vertical slice (A)

```
Open-Meteo OBSERVE
  → VALIDATE
  → EarthState UPDATE
  → BASELINE + MULTI-MODEL FORECAST
  → ENSEMBLE
  → ASSURANCE
  → PUBLISH (API/CLI/Dashboard)
  → Archive OBSERVE (later time)
  → VERIFY
```

## Storage separation

`RAW` · `PROCESSED` · `ASSIMILATED` · `TRAINING` · `VALIDATION` · `FORECAST` · `GROUND_TRUTH` · `MODELS` · `EXPERIMENTS` · `METRICS` · `EVENTS`

## Extensibility

New domains attach via:

1. DataSourceDescriptor in the registry
2. Connector implementing `BaseConnector`
3. Optional specialised engine module
4. Model cards in the model registry

No hard dependency on a single provider.
