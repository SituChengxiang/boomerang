#!/usr/bin/env python3
"""Batch-compute velocity from optimized tracks.

Reads 8 optimized track files under data/*opt.csv (columns: t,x,y,z),
computes velocity components vx/vy/vz, and writes a merged long-format CSV.

Why two methods?
- gradient: pure numerical differentiation (np.gradient); robust to non-uniform dt.
- savgol: local polynomial (Savitzky–Golay) derivative; smoother for noisy data,
		  but assumes roughly-uniform dt.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict, Iterable, List, Tuple

import numpy as np


def load_track(csv_path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	data = np.genfromtxt(csv_path, delimiter=",", names=True)
	if data.ndim == 0:
		raise ValueError(f"CSV must contain at least one row: {csv_path}")
	try:
		t = np.asarray(data["t"], dtype=float)
		x = np.asarray(data["x"], dtype=float)
		y = np.asarray(data["y"], dtype=float)
		z = np.asarray(data["z"], dtype=float)
	except ValueError as exc:
		raise ValueError(f"CSV columns must be named t,x,y,z: {csv_path}") from exc
	order = np.argsort(t)
	return t[order], x[order], y[order], z[order]


def _is_roughly_uniform(t: np.ndarray, rel_tol: float = 1e-3) -> bool:
	if len(t) < 3:
		return True
	dt = np.diff(t)
	med = float(np.median(dt))
	if med <= 0:
		return False
	return float(np.max(np.abs(dt - med)) / med) <= rel_tol


def velocity_gradient(t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	vx = np.gradient(x, t, edge_order=2)
	vy = np.gradient(y, t, edge_order=2)
	vz = np.gradient(z, t, edge_order=2)
	return vx, vy, vz


def _ensure_odd_window(window: int, n: int) -> int:
	w = int(window)
	if w < 3:
		w = 3
	if w > n:
		w = n
	if w % 2 == 0:
		w -= 1
	if w < 3:
		w = 3
	if w > n:
		w = n if n % 2 == 1 else max(1, n - 1)
	return w


def velocity_savgol(
	t: np.ndarray,
	x: np.ndarray,
	y: np.ndarray,
	z: np.ndarray,
	window: int,
	polyorder: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	try:
		from scipy.signal import savgol_filter
	except ImportError as exc:  # pragma: no cover
		raise SystemExit("scipy is required for savgol: pip install scipy") from exc

	n = len(t)
	if n < 3:
		return np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)

	dt = float(np.median(np.diff(t)))
	if dt <= 0:
		raise ValueError("Non-positive dt; cannot compute savgol derivative")

	w = _ensure_odd_window(window, n)
	p = int(polyorder)
	if p >= w:
		p = max(1, w - 2)

	vx = savgol_filter(x, window_length=w, polyorder=p, deriv=1, delta=dt, mode="interp")
	vy = savgol_filter(y, window_length=w, polyorder=p, deriv=1, delta=dt, mode="interp")
	vz = savgol_filter(z, window_length=w, polyorder=p, deriv=1, delta=dt, mode="interp")
	return vx, vy, vz


def discover_opt_tracks(data_dir: pathlib.Path) -> List[pathlib.Path]:
	# Explicit order (the repo has 1,2,3,5,6,7,8,9)
	preferred = [
		"track1opt.csv",
		"track2opt.csv",
		"track3opt.csv",
		"track5opt.csv",
		"track6opt.csv",
		"track7opt.csv",
		"track8opt.csv",
		"track9opt.csv",
	]
	found: List[pathlib.Path] = []
	for name in preferred:
		p = data_dir / name
		if p.exists():
			found.append(p)
	if len(found) == 8:
		return found
	# Fallback: anything matching *opt.csv (sorted)
	return sorted(data_dir.glob("*opt.csv"))


def write_velocity_csv(rows: Iterable[Dict[str, object]], out_path: pathlib.Path) -> None:
	import csv

	out_path.parent.mkdir(parents=True, exist_ok=True)
	rows = list(rows)
	if not rows:
		raise ValueError("No rows to write")

	fieldnames = [
		"track",
		"i",
		"t",
		"x",
		"y",
		"z",
		"vx",
		"vy",
		"vz",
		"speed",
	]
	with out_path.open("w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		w.writeheader()
		for r in rows:
			w.writerow(r)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Compute velocities (vx,vy,vz) from optimized track CSVs and write data/velocity.csv",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("data"), help="Directory containing *opt.csv")
	parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("data/velocity.csv"), help="Output CSV path")
	parser.add_argument("--method", choices=["savgol", "gradient"], default="savgol",
						help="Velocity estimation method")
	parser.add_argument("--savgol-window", type=int, default=11, help="Savgol window length (odd, auto-adjusted)")
	parser.add_argument("--savgol-polyorder", type=int, default=3, help="Savgol polynomial order")
	parser.add_argument("--uniform-rel-tol", type=float, default=1e-3,
						help="If dt non-uniform beyond this (relative), savgol will fall back to gradient")
	args = parser.parse_args()

	tracks = discover_opt_tracks(args.data_dir)
	if not tracks:
		raise SystemExit(f"No *opt.csv found under {args.data_dir}")
	if len(tracks) != 8:
		print(f"Warning: expected 8 opt tracks, found {len(tracks)}")

	all_rows: List[Dict[str, object]] = []
	for csv_path in tracks:
		t, x, y, z = load_track(csv_path)
		track_name = csv_path.stem.replace("opt", "")

		use_method = args.method
		if use_method == "savgol" and not _is_roughly_uniform(t, rel_tol=args.uniform_rel_tol):
			use_method = "gradient"

		if use_method == "savgol":
			vx, vy, vz = velocity_savgol(t, x, y, z, window=args.savgol_window, polyorder=args.savgol_polyorder)
		else:
			vx, vy, vz = velocity_gradient(t, x, y, z)

		speed = np.sqrt(vx * vx + vy * vy + vz * vz)
		for i in range(len(t)):
			all_rows.append({
				"track": track_name,
				"i": int(i),
				"t": float(t[i]),
				"x": float(x[i]),
				"y": float(y[i]),
				"z": float(z[i]),
				"vx": float(vx[i]),
				"vy": float(vy[i]),
				"vz": float(vz[i]),
				"speed": float(speed[i]),
			})

	write_velocity_csv(all_rows, args.output)
	print(f"Wrote: {args.output} (rows={len(all_rows)})")


if __name__ == "__main__":
	main()

