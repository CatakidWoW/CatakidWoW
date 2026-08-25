from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence
from uuid import uuid4

import numpy as np

from quantum_earth.storage import MetadataDB, StorageLayout


def mae(pred: Sequence[float], truth: Sequence[float]) -> float:
    p = np.asarray(pred, dtype=float)
    t = np.asarray(truth, dtype=float)
    mask = ~np.isnan(p) & ~np.isnan(t)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(p[mask] - t[mask])))


def rmse(pred: Sequence[float], truth: Sequence[float]) -> float:
    p = np.asarray(pred, dtype=float)
    t = np.asarray(truth, dtype=float)
    mask = ~np.isnan(p) & ~np.isnan(t)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((p[mask] - t[mask]) ** 2)))


def bias(pred: Sequence[float], truth: Sequence[float]) -> float:
    p = np.asarray(pred, dtype=float)
    t = np.asarray(truth, dtype=float)
    mask = ~np.isnan(p) & ~np.isnan(t)
    if not mask.any():
        return float("nan")
    return float(np.mean(p[mask] - t[mask]))


def correlation(pred: Sequence[float], truth: Sequence[float]) -> float:
    p = np.asarray(pred, dtype=float)
    t = np.asarray(truth, dtype=float)
    mask = ~np.isnan(p) & ~np.isnan(t)
    if mask.sum() < 2:
        return float("nan")
    if np.std(p[mask]) == 0 or np.std(t[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(p[mask], t[mask])[0, 1])


def crps_ensemble(members: np.ndarray, truth: Sequence[float]) -> float:
    """Approximate CRPS for ensemble forecasts (Fair CRPS-style).

    members: shape (n_members, n_times)
    """
    t = np.asarray(truth, dtype=float)
    scores = []
    for i in range(members.shape[1]):
        m = members[:, i]
        m = m[~np.isnan(m)]
        if m.size == 0 or np.isnan(t[i]):
            continue
        term1 = np.mean(np.abs(m - t[i]))
        # pairwise absolute difference
        if m.size > 1:
            term2 = np.mean(np.abs(m[:, None] - m[None, :]))
        else:
            term2 = 0.0
        scores.append(term1 - 0.5 * term2)
    if not scores:
        return float("nan")
    return float(np.mean(scores))


class VerificationEngine:
    def __init__(self, db: MetadataDB | None = None) -> None:
        self.db = db or MetadataDB()
        self.storage = StorageLayout()

    def score_continuous(
        self,
        model_id: str,
        variable: str,
        predictions: Sequence[float],
        truth: Sequence[float],
        members: np.ndarray | None = None,
        meta: dict | None = None,
    ) -> dict:
        result = {
            "id": str(uuid4()),
            "model_id": model_id,
            "variable": variable,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n": int(np.sum(~np.isnan(predictions) & ~np.isnan(truth))),
            "mae": mae(predictions, truth),
            "rmse": rmse(predictions, truth),
            "bias": bias(predictions, truth),
            "correlation": correlation(predictions, truth),
            "crps": crps_ensemble(members, truth) if members is not None else None,
            "meta": meta or {},
        }
        self.db.insert_verification(result["id"], result)
        self.storage.write_json("METRICS", f"{result['id']}.json", result)
        return result
