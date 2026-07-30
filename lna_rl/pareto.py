"""
Multi-objective Pareto front tracker.

Objectives tracked (all maximized internally; NF and Pdc are negated):
  1. -NF_dB       (lower NF = better)
  2.  S21_dB
  3.  IIP3_dBm
  4. -Pdc_mW      (lower power = better)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .simulator import SimResult


@dataclass
class ParetoPoint:
    params: dict
    result: SimResult
    obj: np.ndarray   # 4-element objective vector (all maximized)
    step: int = 0


def _objectives(r: SimResult) -> np.ndarray:
    return np.array([-r.nf_db, r.s21_db, r.iip3_dbm, -r.pdc_mw], dtype=np.float64)


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if a Pareto-dominates b (a >= b in all, a > b in at least one)."""
    return bool(np.all(a >= b) and np.any(a > b))


class ParetoFront:
    def __init__(self):
        self._points: list[ParetoPoint] = []

    def add(self, params: dict, result: SimResult, step: int = 0) -> bool:
        """
        Add a new design point. Returns True if it joined the Pareto front.
        Removes any existing points that are now dominated.
        """
        if not result.valid:
            return False
        obj = _objectives(result)

        # Check if dominated by any existing front member
        for p in self._points:
            if _dominates(p.obj, obj):
                return False

        # Remove any existing points dominated by the new one
        self._points = [p for p in self._points if not _dominates(obj, p.obj)]
        self._points.append(ParetoPoint(params=dict(params), result=result,
                                        obj=obj, step=step))
        return True

    def __len__(self) -> int:
        return len(self._points)

    def best_nf(self) -> Optional[ParetoPoint]:
        if not self._points:
            return None
        return min(self._points, key=lambda p: p.result.nf_db)

    def best_gain(self) -> Optional[ParetoPoint]:
        if not self._points:
            return None
        return max(self._points, key=lambda p: p.result.s21_db)

    def summary(self) -> str:
        if not self._points:
            return "Pareto front: empty"
        lines = [f"Pareto front ({len(self._points)} points):"]
        # Sort by NF ascending
        for p in sorted(self._points, key=lambda x: x.result.nf_db):
            lines.append(
                f"  NF={p.result.nf_db:.2f}dB  S21={p.result.s21_db:.1f}dB  "
                f"IIP3={p.result.iip3_dbm:.1f}dBm  Pdc={p.result.pdc_mw:.1f}mW"
                f"  [step {p.step}]"
            )
        return "\n".join(lines)

    def to_arrays(self) -> tuple[np.ndarray, list[dict]]:
        """Return (N x 4 objective matrix, list of param dicts) for plotting."""
        if not self._points:
            return np.empty((0, 4)), []
        objs = np.stack([p.obj for p in self._points])
        params = [p.params for p in self._points]
        return objs, params

    def hypervolume(self, ref: Optional[np.ndarray] = None) -> float:
        """
        Hypervolume over the first two objectives (-NF, S21), as a single
        scalar for tracking progress during training. Exact staircase sweep,
        which is all that is needed in 2D.

        ref is the worst-acceptable corner and defaults to 10 dB NF at 0 dB
        gain. Points worse than ref on either axis contribute nothing.
        """
        if len(self._points) < 1:
            return 0.0
        if ref is None:
            ref = np.array([-10.0, 0.0])   # -10 = NF of 10 dB, S21 = 0 dB
        pts = sorted(self._points, key=lambda p: p.obj[0])  # sort by -NF ascending
        hv = 0.0
        prev_x = ref[0]
        for p in pts:
            x, y = p.obj[0], p.obj[1]
            if x > prev_x and y > ref[1]:
                hv += (x - prev_x) * (y - ref[1])
                prev_x = x
        return float(hv)
