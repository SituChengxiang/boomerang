#!/usr/bin/env python3
"""Physics verdict layer for trajectory preprocessing.

This module only trims and reports; it never modifies p/v/a values.
Dependencies: numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class VerdictReport:
    valid_duration: float
    reason_for_termination: str
    initial_energy: float
    start_idx: int
    end_idx: int


class PhysicsVerdict:
    """Physical verdict layer for cleaned trajectories."""

    def __init__(
        self,
        mass: float = 0.005,
        energy_tol: float = 0.05,
        sigma_threshold: float = 1.0,
    ) -> None:
        self.mass = float(mass)
        self.energy_tol = float(energy_tol)
        self.sigma_threshold = float(sigma_threshold)

        self._t: Optional[np.ndarray] = None
        self._pos: Optional[np.ndarray] = None
        self._vel: Optional[np.ndarray] = None
        self._acc: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None
        self._start_idx: int = 0
        self._end_idx: int = 0
        self._reason: str = ""
        self._energy: Optional[np.ndarray] = None

    def _extract_inputs(self, data: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Extract t, pos, vel, acc, sigma from dict/EstimatorOutput/DataFrame-like."""
        if isinstance(data, dict):
            t = data.get("t") or data.get("time")
            pos = data.get("pos") or data.get("p")
            vel = data.get("vel") or data.get("v")
            acc = data.get("acc") or data.get("a")
            sigma = data.get("sigma")
        else:
            # EstimatorOutput-like (attribute access)
            t = getattr(data, "t_std", None) or getattr(data, "t", None)
            pos = getattr(data, "pos", None)
            vel = getattr(data, "vel", None)
            acc = getattr(data, "acc", None)
            sigma = getattr(data, "sigma", None)

        if t is None or pos is None or vel is None or acc is None:
            raise ValueError("Input must provide time, pos, vel, acc (and optional sigma)")

        t = np.asarray(t, dtype=float).ravel()
        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float)
        acc = np.asarray(acc, dtype=float)
        sigma = None if sigma is None else np.asarray(sigma, dtype=float)

        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("pos must have shape (N, 3)")
        if vel.ndim != 2 or vel.shape[1] != 3:
            raise ValueError("vel must have shape (N, 3)")
        if acc.ndim != 2 or acc.shape[1] != 3:
            raise ValueError("acc must have shape (N, 3)")
        if t.size != pos.shape[0] or t.size != vel.shape[0] or t.size != acc.shape[0]:
            raise ValueError("time length must match pos/vel/acc length")

        return t, pos, vel, acc, sigma

    def calculate_energies(self, t: np.ndarray, pos: np.ndarray, vel: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute kinetic, potential, and total energy using provided velocities."""
        v2 = np.sum(vel * vel, axis=1)
        ek = 0.5 * self.mass * v2
        ep = self.mass * 9.793 * pos[:, 2]
        et = ek + ep
        dE_dt = np.gradient(et, t, edge_order=2 if t.size >= 3 else 1)
        return {"Ek": ek, "Ep": ep, "E": et, "dE_dt": dE_dt}

    def diagnose_launch(self, t: np.ndarray, energy: np.ndarray) -> int:
        """Suggest start index if launch energy is unstable in first 0.2s."""
        if t.size < 2:
            return 0
        t_ref = 0.1
        idx_ref = int(np.argmin(np.abs(t - t_ref)))
        e_ref = float(energy[idx_ref])
        e0 = float(energy[0])
        scale = max(abs(e_ref), 1.0)
        if abs(e0 - e_ref) / scale <= self.energy_tol:
            return 0

        # Find earliest index within first 0.2s that stabilizes
        cutoff = float(t[0] + 0.2)
        window = np.where(t <= cutoff)[0]
        if window.size == 0:
            return 0
        for i in window:
            if abs(energy[i] - e_ref) / scale <= self.energy_tol:
                return int(i)
        return int(window[-1])

    def determine_valid_range(self, t: np.ndarray, pos: np.ndarray, vel: np.ndarray, sigma: Optional[np.ndarray]) -> Tuple[int, int, str]:
        """Determine valid [start_idx, end_idx) based on physical rules."""
        energies = self.calculate_energies(t, pos, vel)
        et = energies["E"]
        dE_dt = energies["dE_dt"]

        start_idx = self.diagnose_launch(t, et)
        end_idx = t.size
        reason = "OK"

        # Rule A: Energy rebound (positive dE/dt) sustained
        rebound_mask = dE_dt > 0
        if rebound_mask.any():
            streak = 0
            for i in range(start_idx, t.size):
                if rebound_mask[i] and (et[i] - et[start_idx]) > self.energy_tol * max(abs(et[start_idx]), 1.0):
                    streak += 1
                    if streak >= 3:
                        end_idx = i
                        reason = "Energy Spike"
                        break
                else:
                    streak = 0

        # Rule B: Uncertainty (sigma)
        if end_idx == t.size and sigma is not None:
            if sigma.ndim == 2:
                sigma_level = np.max(sigma, axis=1)
            else:
                sigma_level = sigma
            bad = np.where(sigma_level > self.sigma_threshold)[0]
            if bad.size > 0:
                end_idx = int(bad[0])
                reason = "High Uncertainty"

        return start_idx, max(end_idx, start_idx + 1), reason

    def evaluate(self, data: Any) -> VerdictReport:
        """Run verdict and store internal state."""
        t, pos, vel, acc, sigma = self._extract_inputs(data)
        self._t = t
        self._pos = pos
        self._vel = vel
        self._acc = acc
        self._sigma = sigma

        start_idx, end_idx, reason = self.determine_valid_range(t, pos, vel, sigma)
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._reason = reason

        energies = self.calculate_energies(t, pos, vel)
        self._energy = energies["E"]

        valid_duration = float(t[end_idx - 1] - t[start_idx]) if end_idx > start_idx else 0.0
        initial_energy = float(energies["E"][start_idx])

        return VerdictReport(
            valid_duration=valid_duration,
            reason_for_termination=reason,
            initial_energy=initial_energy,
            start_idx=start_idx,
            end_idx=end_idx,
        )

    def get_clean_trajectory(self) -> Dict[str, np.ndarray]:
        """Return trimmed trajectory without modifying values."""
        if self._t is None or self._pos is None or self._vel is None or self._acc is None:
            raise RuntimeError("Call evaluate() before get_clean_trajectory().")

        sl = slice(self._start_idx, self._end_idx)
        out = {
            "t": self._t[sl],
            "pos": self._pos[sl],
            "vel": self._vel[sl],
            "acc": self._acc[sl],
        }
        if self._sigma is not None:
            out["sigma"] = self._sigma[sl]
        return out

    def get_verdict_report(self) -> Dict[str, float | str | int]:
        """Return verdict report as dict."""
        if self._energy is None:
            raise RuntimeError("Call evaluate() before get_verdict_report().")

        valid_duration = float(self._t[self._end_idx - 1] - self._t[self._start_idx]) # pyright: ignore[reportOptionalSubscript]
        return {
            "valid_duration": valid_duration,
            "reason_for_termination": self._reason,
            "initial_energy": float(self._energy[self._start_idx]),
            "start_idx": int(self._start_idx),
            "end_idx": int(self._end_idx),
        }


__all__ = ["PhysicsVerdict", "VerdictReport"]
