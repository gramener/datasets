#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2.2",
#     "numpy>=1.26",
# ]
# ///
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Hypotheses baked into the synthetic data:
# 1) Over-target temperatures (>78C) materially increase defect and scrap rates, especially when humidity is high.
# 2) Overdue maintenance (>30 days) raises vibration and downtime while reducing throughput.
# 3) Expert operators keep defect and scrap rates lower and reduce downtime versus novice operators.
# 4) Night shift experiences higher scrap and downtime than day shift.
# 5) High vibration readings correlate with both downtime and defect rates.


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value within low/high bounds."""
    return max(low, min(value, high))


def generate_rows(n_rows: int, seed: int = 7) -> pd.DataFrame:
    """Generate manufacturing instrumentation log rows."""
    rng = np.random.default_rng(seed)
    start_time = datetime.now() - timedelta(days=45)

    lines = ["L1", "L2", "L3", "L4"]
    products = ["P-ABS", "P-NYLON", "P-PEEK", "P-PP"]
    operators = [("OP-1", "expert"), ("OP-2", "intermediate"), ("OP-3", "novice"), ("OP-4", "intermediate"), ("OP-5", "expert")]

    rows: list[dict[str, float | str | bool | datetime]] = []
    for _ in range(n_rows):
        timestamp = start_time + timedelta(minutes=int(rng.integers(0, 45 * 24 * 60)))
        shift = rng.choice(["Day", "Swing", "Night"], p=[0.45, 0.35, 0.20])
        line_id = rng.choice(lines)
        machine_id = f"{line_id}-M{rng.integers(1, 7)}"
        product_code = rng.choice(products, p=[0.32, 0.26, 0.21, 0.21])
        batch_id = f"B{timestamp:%y%m%d}-{rng.integers(100, 999)}"
        operator_id, operator_experience = operators[rng.integers(0, len(operators))]

        days_since_maintenance = clamp(float(rng.normal(18, 12)), 0.0, 60.0)
        if days_since_maintenance > 30:
            maintenance_status = "overdue"
        elif days_since_maintenance > 23:
            maintenance_status = "due_soon"
        else:
            maintenance_status = "ok"

        target_temp_c = rng.uniform(72, 76)
        high_temp_spike = rng.random() < (0.18 + 0.20 * (maintenance_status == "overdue"))
        actual_temp_c = target_temp_c + rng.normal(0, 1.2) + (rng.normal(4.0, 0.8) if high_temp_spike else 0.0)

        humidity_percent = clamp(float(rng.normal(48 + (shift == "Night") * 4, 8)), 28.0, 82.0)
        pressure_bar = clamp(float(rng.normal(5.5, 0.3)), 4.6, 6.4)
        vibration_mm_s = clamp(
            float(rng.normal(3.2, 0.5) + 0.05 * max(days_since_maintenance - 18, 0) + (0.8 if maintenance_status == "overdue" else 0.0)),
            1.5,
            7.0,
        )

        base_speed = rng.normal(120, 12)
        speed_penalty = 0.0
        speed_penalty += 8 if maintenance_status == "overdue" else 4 if maintenance_status == "due_soon" else 0
        speed_penalty += 6 if actual_temp_c > 80 else 0
        speed_penalty += 4 if humidity_percent > 65 else 0
        speed_penalty += 6 if operator_experience == "novice" else -5 if operator_experience == "expert" else 0
        line_speed_units_min = clamp(float(base_speed - speed_penalty), 80.0, 155.0)

        window_minutes = int(rng.integers(8, 16))
        throughput_units = int(line_speed_units_min * window_minutes)

        defect_rate_percent = (
            1.5
            + 0.35 * max(actual_temp_c - 75, 0)
            + 0.06 * max(humidity_percent - 55, 0)
            + (0.60 if maintenance_status == "overdue" else 0.25 if maintenance_status == "due_soon" else 0.0)
            + 0.40 * max(vibration_mm_s - 4.0, 0)
            + (0.60 if shift == "Night" else 0.25 if shift == "Swing" else 0.0)
            - (0.70 if operator_experience == "expert" else 0.20 if operator_experience == "intermediate" else -0.50)
            + rng.normal(0, 0.25)
        )
        defect_rate_percent = clamp(defect_rate_percent, 0.2, 15.0)

        avg_unit_weight_kg = rng.uniform(0.6, 1.2)
        scrap_kg = max(0.0, throughput_units * (defect_rate_percent / 100) * avg_unit_weight_kg)

        downtime_minutes = max(
            0.0,
            rng.normal(2.2, 1.0)
            + (6.0 if maintenance_status == "overdue" else 3.0 if maintenance_status == "due_soon" else 1.0)
            + 1.2 * max(vibration_mm_s - 4.0, 0)
            + (1.5 if shift == "Night" else 0.5 if shift == "Swing" else 0.0)
            + (1.8 if actual_temp_c > 80 else 0.0)
            + (0.8 if defect_rate_percent > 5 else 0.0),
        )

        energy_kwh = clamp(float(32 + 0.16 * line_speed_units_min * window_minutes / 10 + 0.35 * (actual_temp_c - 70)), 10.0, 140.0)
        ambient_temp_c = clamp(float(rng.normal(24 + (shift == "Day") * 2, 2.5)), 18.0, 32.0)
        quality_alert = bool((actual_temp_c > 80) or (vibration_mm_s > 5.2) or (defect_rate_percent > 6.0))

        rows.append(
            {
                "timestamp": timestamp.isoformat(),
                "shift": shift,
                "line_id": line_id,
                "machine_id": machine_id,
                "product_code": product_code,
                "batch_id": batch_id,
                "operator_id": operator_id,
                "operator_experience": operator_experience,
                "days_since_maintenance": round(days_since_maintenance, 1),
                "maintenance_status": maintenance_status,
                "target_temp_c": round(target_temp_c, 1),
                "actual_temp_c": round(actual_temp_c, 1),
                "humidity_percent": round(humidity_percent, 1),
                "pressure_bar": round(pressure_bar, 2),
                "vibration_mm_s": round(vibration_mm_s, 2),
                "line_speed_units_min": round(line_speed_units_min, 1),
                "window_minutes": window_minutes,
                "throughput_units": throughput_units,
                "defect_rate_percent": round(defect_rate_percent, 2),
                "scrap_kg": round(scrap_kg, 2),
                "downtime_minutes": round(downtime_minutes, 2),
                "energy_kwh": round(energy_kwh, 2),
                "ambient_temp_c": round(ambient_temp_c, 1),
                "quality_alert": quality_alert,
            }
        )

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    """Generate and store synthetic manufacturing instrumentation logs."""
    n_rows = 2000
    df = generate_rows(n_rows=n_rows)
    output_path = Path(__file__).with_name("manufacturing_instrumentation.csv")
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df):,} rows to {output_path}")


if __name__ == "__main__":
    main()
