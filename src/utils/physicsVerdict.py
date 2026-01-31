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
        mass: float = 0.002183,
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

    @staticmethod
    def _extract_inputs(
            data: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Extract t, pos, vel, acc, sigma from dict/EstimatorOutput/DataFrame-like."""
        if isinstance(data, dict):
            t = data.get("t") or data.get("time")
            pos = data.get("pos") or data.get("p")
            vel = data.get("vel") or data.get("v")
            acc = data.get("acc") or data.get("a")
            sigma = data.get("sigma")
        else:
            # EstimatorOutput-like (attribute access)
            # Avoid 'or' operator with numpy arrays (ambiguous truth value)
            t = getattr(data, "t_std", None)
            if t is None:
                t = getattr(data, "t", None)
            pos = getattr(data, "pos", None)
            vel = getattr(data, "vel", None)
            acc = getattr(data, "acc", None)
            sigma = getattr(data, "sigma", None)

        if t is None or pos is None or vel is None or acc is None:
            raise ValueError(
                "Input must provide time, pos, vel, acc (and optional sigma)"
            )

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

    def calculate_energies(
        self, t: np.ndarray, pos: np.ndarray, vel: np.ndarray
    ) -> Dict[str, np.ndarray]:
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

        # Find the earliest index within first 0.2s that stabilizes
        cutoff = float(t[0] + 0.2)
        window = np.where(t <= cutoff)[0]
        if window.size == 0:
            return 0
        for i in window:
            if abs(energy[i] - e_ref) / scale <= self.energy_tol:
                return int(i)
        return int(window[-1])

    def determine_valid_range(
        self,
        t: np.ndarray,
        pos: np.ndarray,
        vel: np.ndarray,
        sigma: Optional[np.ndarray],
    ) -> Tuple[int, int, str]:
        """Determine valid [start_idx, end_idx) based on dE/dt power analysis.

        Assumption: The input trajectory is already roughly confined to "release" to "catch" phase.
        We trim specific outliers where dE/dt exceeds physical limits caused by tracker noise.
        """
        energies = self.calculate_energies(t, pos, vel)
        # Calculate specific power [W/kg] = dE/dt
        power = energies["dE_dt"]
        abs_power = np.abs(power)

        # Adaptive thresholding:
        # Use median absolute power as a baseline for "normal" aerodynamic work.
        # Spikes > 4x Median are likely tracker noise (non-physical).
        median_power = float(np.median(abs_power))
        # Ensure a minimal floor for very smooth flights to avoid over-trimming
        # Lowered to 0.5 W/kg based on observed data (mean ~0.2, max ~0.95)
        baseline = max(median_power, 0.5)
        power_threshold = 3.0 * baseline

        # --- Start Trimming ---
        # Scan from start: aggressive trim if initial power is huge (launch artifacts)
        start_idx = 0
        for i in range(min(len(t) // 3, 20)):  # Only check first few frames
            if abs_power[i] > power_threshold:
                start_idx = i + 1
            else:
                # Once we find a good frame, assume launch artifact is over
                # Check 2 more frames ahead to be sure
                if (
                    i + 2 < len(t)
                    and abs_power[i + 1] < power_threshold
                    and abs_power[i + 2] < power_threshold
                ):
                    start_idx = i
                    break

        # --- End Trimming ---
        # Scan from end: trim backward if final power is huge (impact/catch artifacts)
        end_idx = t.size
        for i in range(t.size - 1, max(t.size - 20, start_idx + 10), -1):
            if abs_power[i] > power_threshold:
                end_idx = i
            else:
                # Found a valid frame from the back
                if (
                    i - 2 > start_idx
                    and abs_power[i - 1] < power_threshold
                    and abs_power[i - 2] < power_threshold
                ):
                    end_idx = i + 1  # Include this valid frame
                    break

        reason = "OK"
        if start_idx > 0:
            reason = "Launch Artifact Trimmed"
        if end_idx < t.size:
            reason += " & End Spike Trimmed" if start_idx > 0 else "End Spike Trimmed"

        # Rule B: Uncertainty (sigma) - fallback
        if sigma is not None:
            if sigma.ndim == 2:
                sigma_level = np.max(sigma, axis=1)
            else:
                sigma_level = sigma
            # Only trim from the end based on sigma
            valid_mask = sigma_level[:end_idx] <= self.sigma_threshold
            # Find last valid index
            if not valid_mask.all():
                # Find the first violation after start_idx
                bad_indices = np.where(~valid_mask)[0]
                bad_after_start = bad_indices[bad_indices >= start_idx]
                if bad_after_start.size > 0:
                    new_end = int(bad_after_start[0])
                    if new_end < end_idx:
                        end_idx = new_end
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

        valid_duration = (
            float(t[end_idx - 1] - t[start_idx]) if end_idx > start_idx else 0.0
        )
        initial_energy = float(energies["E"][start_idx])

        return VerdictReport(
            valid_duration=valid_duration,
            reason_for_termination=reason,
            initial_energy=initial_energy,
            start_idx=start_idx,
            end_idx=end_idx,
        )

    def get_clean_trajectory(self) -> dict[str, type[None[Any]]]: # pyright: ignore[reportInvalidTypeArguments]
        """Return trimmed trajectory without modifying values."""
        if (
            self._t is None
            or self._pos is None
            or self._vel is None
            or self._acc is None
        ):
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
        return out # pyright: ignore[reportReturnType]

    def get_verdict_report(self) -> Dict[str, float | str | int]:
        """Return verdict report as dict."""
        if self._energy is None:
            raise RuntimeError("Call evaluate() before get_verdict_report().")

        valid_duration = float(self._t[self._end_idx - 1] - self._t[self._start_idx])  # pyright: ignore[reportOptionalSubscript]
        return {
            "valid_duration": valid_duration,
            "reason_for_termination": self._reason,
            "initial_energy": float(self._energy[self._start_idx]),
            "start_idx": int(self._start_idx),
            "end_idx": int(self._end_idx),
        }


__all__ = ["PhysicsVerdict", "VerdictReport"]
