# Implementation Plan

## Inspection summary (2026-08-25)

| Item | Finding |
|------|---------|
| Repository | Empty greenfield (`README.md` only) |
| Hardware | 4 CPU, 15 GiB RAM, **no GPU** |
| Python | 3.12.3 |
| Node | 22.14.0 (unused in Stage 1 backend) |
| Reusable infra | None |
| Conflicts | None |
| External data | Open-Meteo forecast + archive reachable; NOAA/DWD reachable |

## Stage roadmap

Follow the Stage 1–10 strategy from the product brief. Do not expand a stage until: RUN TESTS → BENCHMARK → DOCUMENT → VERIFY.

### Stage 1 (this PR) — foundations

- Repository architecture, configuration
- Data-source registry
- Observation + EarthState schemas + provenance
- Model + experiment registries
- Verification framework
- System health + Quantum Assurance

### Vertical Slice A (this PR) — prove the loop with real data

- Open-Meteo weather connector (free)
- Earth-state generation for temperature / precipitation / wind
- Persistence, climatology, AR(1) baselines
- Multi-model ensemble (ECMWF / GFS / ICON via Open-Meteo)
- Short-range forecast API / CLI / dashboard
- Historical archive verification + benchmark report

### Next slices (not this PR)

- Stage 2+: additional free connectors (METAR/ISD, satellite products when licences/APIs confirmed)
- Stage 3+: richer short-range models
- Stage 4+: nowcast / radar where open
- Later: hydrology, cryosphere, phenology, fire, ocean, AQ, space weather, geohazards

## Blocked external dependencies

| Component | Status | Notes |
|-----------|--------|-------|
| Commercial satellite feeds | BLOCKED_EXTERNAL_DEPENDENCY | Not required for Stage 1 |
| Proprietary NWP direct GRIB pipelines | BLOCKED_EXTERNAL_DEPENDENCY | Open-Meteo used as free aggregation |
| GPU training cluster | Unavailable | Reduced-capability CPU mode |

## Success criteria for this PR

1. Real observations ingested from Open-Meteo
2. EarthState values carry uncertainty + provenance
3. Probabilistic ensemble forecasts produced
4. Verification metrics computed against archive ground truth
5. Assurance never fabricates certainty
6. Tests pass including adversarial corruption / missing-source cases
7. Docs state exactly what is real vs simulated
