#!/usr/bin/env python3
"""Generate synthetic EV-like battery telemetry for field workflow demos."""

from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate demo EV battery telemetry.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--n-cells", type=int, default=8, help="Number of cells in fleet.")
    parser.add_argument("--cycles-per-cell", type=int, default=20, help="Cycles per cell.")
    parser.add_argument("--points-per-cycle", type=int, default=120, help="Telemetry points per cycle.")
    parser.add_argument("--time-step", type=float, default=12.0, help="Seconds between points.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--anomaly-fraction", type=float, default=0.008, help="Approximate anomaly probability.")
    parser.add_argument("--soh-capacity-ah", type=float, default=2.2, help="Initial capacity proxy in Ah.")
    return parser.parse_args()


@dataclass
class CellConfig:
    base_temp: float
    base_r: float
    current_scale: float
    noise: float


def _cell_config(cell_index: int, rng: np.random.Generator) -> CellConfig:
    return CellConfig(
        base_temp=22.0 + 8.0 * np.sin(cell_index / 3),
        base_r=0.055 + 0.002 * cell_index + 0.003 * rng.normal(),
        current_scale=35.0 + 4.0 * rng.normal(),
        noise=0.35 + 0.05 * rng.random(),
    )


def _simulate_cycle(
    t0: float,
    cell: str,
    cycle_idx: int,
    points: int,
    dt: float,
    cfg: CellConfig,
    capacity_ref: float,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    fatigue = cycle_idx / 80.0
    for step in range(points):
        frac = step / max(1, points - 1)
        timestamp = t0 + step * dt
        soc_base = 0.92 - cycle_idx * 0.012
        soc = max(0.08, soc_base - 0.58 * frac + 0.01 * np.sin((step + cycle_idx) / 9.0))
        disch = step < int(points * 0.65)
        phase_current = -cfg.current_scale * (1.0 + 0.6 * fatigue) if disch else cfg.current_scale * 0.85
        phase_current += 2.5 * np.sin(step / 12.0) + cfg.noise * rng.normal()
        ocv = 3.25 + 1.15 * soc + 0.02 * np.cos(cycle_idx / 6.0)
        resistance = cfg.base_r * (1.0 + 0.8 * fatigue + 0.02 * np.sin(step / 16.0))
        voltage = ocv - resistance * phase_current + 0.01 * rng.normal()
        temperature = cfg.base_temp + 0.45 * cycle_idx + 6.0 * frac + 1.5 * np.cos(step / 10.0) + 0.08 * phase_current
        capacity = capacity_ref * (1.0 - 0.0045 * cycle_idx) + cfg.noise * rng.normal()

        rows.append(
            {
                "timestamp": float(timestamp),
                "cell_id": cell,
                "cycle_id": int(cycle_idx),
                "current": float(phase_current),
                "voltage": float(voltage),
                "temperature": float(temperature),
                "soc": float(np.clip(soc, 0.0, 1.0)),
                "capacity": float(max(0.45, capacity)),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float]] = []
    cursor = 0.0
    t0 = 1_700_000_000.0
    for ci in range(args.n_cells):
        cfg = _cell_config(ci, rng)
        for cycle in range(args.cycles_per_cell):
            cycle_rows = _simulate_cycle(
                cursor + t0,
                f"cell-{ci:03d}",
                cycle,
                args.points_per_cycle,
                args.time_step,
                cfg,
                args.soh_capacity_ah,
                rng,
            )
            rows.extend(cycle_rows)
            cursor += args.points_per_cycle * args.time_step

    df = pd.DataFrame(rows).sort_values(["cell_id", "cycle_id", "timestamp"]).reset_index(drop=True)

    if args.anomaly_fraction > 0:
        n = len(df)
        n_anom = max(1, int(n * args.anomaly_fraction))
        anom_idx = rng.choice(n, size=n_anom, replace=False)
        for idx in anom_idx:
            if rng.random() < 0.6:
                df.loc[int(idx), "voltage"] *= 1.0 + 0.25 * rng.normal()
            else:
                df.loc[int(idx), "temperature"] += 35.0 * np.sign(rng.normal())
            if rng.random() < 0.3:
                dup_idx = max(0, int(idx) - 2)
                df.loc[int(idx), "timestamp"] = df.loc[dup_idx, "timestamp"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote demo telemetry with {len(df)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
