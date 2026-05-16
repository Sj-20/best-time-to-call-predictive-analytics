# -*- coding: utf-8 -*-
"""
Synthetic outbound-call data generator for the Best Time To Call (BTC) project.

This replaces confidential client data with realistic synthetic data using the same
core fields from the original descriptive analytics script:
- Original Call Date
- Original Call Hour
- Original Call Minute
- Total Duration
- Total Billsec
- Outbound Call
- Uniqueid
- Callerid
- Last Event
- Disposition
- Agent Name

The synthetic labels are intentionally generated with business logic:
connection probability is higher during certain hours, varies by weekday/weekend,
month, and agent group. This allows the predictive model to learn meaningful
patterns without exposing any real customer, client, or call records.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


WORKING_HOURS = list(range(8, 20))
AGENTS = [f"t1.tlp.tso.ms.agent_{i:02d}" for i in range(1, 31)] + [f"external.agent_{i:02d}" for i in range(1, 6)]


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1 / (1 + np.exp(-x))


def generate_synthetic_btc_data(
    n_rows: int = 120_000,
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a realistic synthetic outbound-call dataset."""
    rng = np.random.default_rng(random_state)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Slightly higher call volume on weekdays than weekends.
    date_weights = np.array([1.25 if d.weekday() < 5 else 0.65 for d in dates], dtype=float)
    date_weights = date_weights / date_weights.sum()
    call_dates = rng.choice(dates, size=n_rows, p=date_weights)

    # Call-center working-hour distribution: more attempts late morning and afternoon.
    hour_weights = np.array([0.055, 0.070, 0.095, 0.105, 0.090, 0.085, 0.090, 0.105, 0.110, 0.095, 0.060, 0.040])
    hours = rng.choice(WORKING_HOURS, size=n_rows, p=hour_weights / hour_weights.sum())
    minutes = rng.integers(0, 60, size=n_rows)

    caller_pool = [f"CUST{str(i).zfill(7)}" for i in range(1, int(n_rows * 0.72) + 1)]
    caller_ids = rng.choice(caller_pool, size=n_rows, replace=True)
    agents = rng.choice(AGENTS, size=n_rows, replace=True)

    call_dates_series = pd.Series(pd.to_datetime(call_dates))
    dow = call_dates_series.dt.weekday.to_numpy()
    month = call_dates_series.dt.month.to_numpy()
    is_weekend = (dow >= 5).astype(int)

    # Business pattern: connection rates tend to be stronger late morning and late afternoon.
    hour_effect = (
        0.65 * np.isin(hours, [10, 11]).astype(float)
        + 0.50 * np.isin(hours, [16, 17]).astype(float)
        + 0.25 * np.isin(hours, [14, 15]).astype(float)
        - 0.55 * np.isin(hours, [8, 19]).astype(float)
    )
    weekday_effect = np.where(is_weekend == 1, -0.30, 0.12)
    friday_effect = np.where(dow == 4, 0.10, 0.00)
    seasonal_effect = np.where(np.isin(month, [11, 12]), 0.20, 0.00) + np.where(np.isin(month, [1, 2]), -0.10, 0.00)
    minute_effect = np.where((minutes >= 0) & (minutes <= 10), 0.08, 0.00)
    agent_effect = np.where(pd.Series(agents).str.contains("t1.tlp.tso.ms").to_numpy(), 0.22, -0.35)

    logit = -1.95 + hour_effect + weekday_effect + friday_effect + seasonal_effect + minute_effect + agent_effect
    connect_probability = sigmoid(logit)
    connected = rng.binomial(1, connect_probability, size=n_rows)

    # Outcome fields are generated after the synthetic connection flag.
    disposition = np.where(
        connected == 1,
        "ANSWERED",
        rng.choice(["NO ANSWER", "BUSY", "FAILED", "CANCEL"], size=n_rows, p=[0.58, 0.18, 0.14, 0.10]),
    )
    total_billsec = np.where(
        connected == 1,
        rng.integers(20, 420, size=n_rows),
        rng.integers(0, 19, size=n_rows),
    )
    total_duration = total_billsec + rng.integers(5, 60, size=n_rows)
    last_event = np.where(connected == 1, "Hangup", rng.choice(["DialEnd", "Hangup", "Congestion"], size=n_rows))

    df = pd.DataFrame(
        {
            "Original Call Date": call_dates_series.dt.date,
            "Original Call Hour": hours,
            "Original Call Minute": minutes,
            "Total Duration": total_duration,
            "Total Billsec": total_billsec,
            "Outbound Call": "Y",
            "Uniqueid": [f"UID{random_state}{i:010d}" for i in range(n_rows)],
            "Callerid": caller_ids,
            "Last Event": last_event,
            "Disposition": disposition,
            "Agent Name": agents,
        }
    )

    df = df.sort_values(["Original Call Date", "Original Call Hour", "Original Call Minute", "Uniqueid"]).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=120_000)
    parser.add_argument("--start-date", type=str, default="2025-01-01")
    parser.add_argument("--end-date", type=str, default="2025-12-31")
    parser.add_argument("--output", type=str, default="data/synthetic_btc_outbound_calls.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_btc_data(args.rows, args.start_date, args.end_date, args.seed)
    df.to_csv(output_path, index=False)
    print(f"Saved synthetic dataset: {output_path.resolve()} | rows={len(df):,}")


if __name__ == "__main__":
    main()
