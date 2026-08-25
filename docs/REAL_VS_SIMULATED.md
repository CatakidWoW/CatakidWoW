# Real vs Simulated / Blocked

This document is mandatory scientific integrity reporting for QUANTUM EARTH Stage 1.

## REAL (obtained and used)

| Capability | Evidence |
|------------|----------|
| Open-Meteo forecast API ingest | Live HTTP to `api.open-meteo.com`; raw JSON archived under `data/RAW/open-meteo-forecast/` |
| Open-Meteo archive / reanalysis ingest | Live HTTP to `archive-api.open-meteo.com`; raw JSON under `data/RAW/open-meteo-archive/` |
| Multi-model members ECMWF/GFS/ICON | Returned by Open-Meteo `models=` parameter |
| Observation validation | Physical-range checks, dedupe, corruption rejection |
| EarthState quality-weighted blend | Computed locally from validated observations |
| Baseline persistence / climatology / AR(1) | Implemented and scored against archive truth |
| Ensemble quantiles + precip threshold probs | Computed from member matrix |
| Verification MAE/RMSE/bias/corr/CRPS | Against archive ground truth (not synthetic) |
| Quantum Assurance GREEN/YELLOW/RED | Heuristic but explicit; reasons attached |
| API / CLI / dashboard | Serve real loop outputs |

## NOT CLAIMED / BLOCKED

| Component | Status |
|-----------|--------|
| Direct NOAA GFS GRIB ingest | `BLOCKED_EXTERNAL_DEPENDENCY` (reachable, not wired) |
| DWD Open Data connector | `BLOCKED_EXTERNAL_DEPENDENCY` |
| GOES / satellite imagery pipeline | `BLOCKED_EXTERNAL_DEPENDENCY` — no fake satellite integration |
| Radar nowcasting | Not implemented |
| Hydrology / flood engine | Not implemented |
| Phenology / vegetation engine | Not implemented |
| Fire / ocean / AQ / space weather / geohazards | Not implemented |
| Event catalogue lifecycle | Explicit `NOT_IMPLEMENTED` / `UNKNOWN` |
| Autonomous research loop | Skeleton only |
| GPU training | Hardware unavailable — reduced-capability CPU mode |
| Kalman / EnKF / 4D-Var | Interfaces stubbed only |

## NEVER DONE

- Fabricated API responses
- Synthetic observations silently substituted for real ones
- LLM-generated probabilities presented as forecasts
- Deterministic earthquake prediction claims
- Presenting experimental models as production without metrics

## Integrity rule

If a response cannot be traced to DATA → MODEL → COMPUTATION → METRIC → RESULT, it must be labelled `UNKNOWN` or `HYPOTHESIS`.
