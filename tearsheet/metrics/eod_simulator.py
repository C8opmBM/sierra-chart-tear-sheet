"""Prop-firm evaluation pass/fail simulator.

Models the common "EOD trailing drawdown" account structure (e.g. Lucid
Trading's LucidFlex/LucidPro): a drawdown floor that trails your highest
end-of-day balance upward, locks once you clear a starting_balance +
drawdown_limit buffer, plus an independent daily loss limit, checked against
a profit target.

Known simplifications (not modeled): the "no single day > X% of total
evaluation profit" consistency rule, and any minimum-trading-days
requirement. Both can meaningfully change pass/fail outcomes near the
target and are left for a future iteration.
"""

from __future__ import annotations

import random
from typing import Any

import pandas as pd


def daily_pnls_from_trades(trades: list[dict[str, Any]], key: str = "net_pnl") -> list[float]:
    """Aggregate trade P&L into one value per calendar day (by ``exit_time``).

    Returns values in chronological day order. Trades without an
    ``exit_time`` (still open) are excluded.
    """
    by_day: dict[Any, float] = {}
    for t in trades:
        et = t.get("exit_time")
        if et is None:
            continue
        day = pd.Timestamp(et).date()
        by_day[day] = by_day.get(day, 0.0) + float(t.get(key, 0.0))
    return [by_day[d] for d in sorted(by_day)]


def _run_single_path(
    daily_pnls: list[float],
    starting_balance: float,
    drawdown_limit: float,
    daily_loss_limit: float | None,
    profit_target: float | None,
) -> str:
    """Replay one sequence of daily P&Ls through the EOD state machine.

    Returns one of ``"pass"``, ``"fail_drawdown"``, ``"fail_daily_loss"``, or
    ``"undetermined"`` (ran out of days without resolving either way).
    """
    balance = starting_balance
    peak_eod = starting_balance
    # Once peak_eod reaches this point, the trailing floor locks at
    # starting_balance and stops rising further.
    lock_point = starting_balance + drawdown_limit

    for pnl in daily_pnls:
        # Daily loss limit is checked against the day's own P&L, independent
        # of trailing-drawdown state — a single bad day fails immediately
        # even if the account is otherwise well clear of its floor.
        if daily_loss_limit is not None and pnl < -daily_loss_limit:
            return "fail_daily_loss"

        balance += pnl
        peak_eod = max(peak_eod, balance)
        floor = min(peak_eod, lock_point) - drawdown_limit

        if balance < floor:
            return "fail_drawdown"

        if profit_target is not None and balance >= starting_balance + profit_target:
            return "pass"

    return "undetermined"


def simulate_evaluation(
    trades: list[dict[str, Any]],
    starting_balance: float,
    drawdown_limit: float,
    daily_loss_limit: float | None = None,
    profit_target: float | None = None,
    n_sims: int = 2000,
    max_days_multiplier: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap-resample trading days through the EOD evaluation rules.

    Parameters
    ----------
    trades:
        Trade dicts with ``exit_time`` and ``net_pnl`` (as returned by
        :func:`tearsheet.recon.trades.reconstruct_trades`).
    starting_balance:
        Account balance at evaluation start.
    drawdown_limit:
        Max trailing drawdown in dollars (e.g. 2000 for a $2,000 trailing
        max loss).
    daily_loss_limit:
        Optional. Max single-day loss in dollars before automatic failure.
    profit_target:
        Optional. Dollar profit above starting_balance needed to pass. If
        omitted, "pass" can never trigger and every path resolves to
        fail_drawdown/fail_daily_loss/undetermined.
    n_sims:
        Number of bootstrap simulations.
    max_days_multiplier:
        Each simulated path samples ``n_real_days * max_days_multiplier``
        days (with replacement) before being marked "undetermined" if
        neither pass nor fail has occurred — a practical cap standing in for
        these evaluations' lack of a hard time limit.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict with keys ``stats`` (probabilities and the historical actual
    outcome) and ``daily_pnls`` (the real day-by-day P&L series used).
    Returns ``{"stats": {}, "daily_pnls": []}`` if there's no usable data.
    """
    daily_pnls = daily_pnls_from_trades(trades)
    n_real_days = len(daily_pnls)

    if n_real_days == 0 or starting_balance <= 0 or drawdown_limit <= 0:
        return {"stats": {}, "daily_pnls": daily_pnls}

    actual_outcome = _run_single_path(
        daily_pnls, starting_balance, drawdown_limit, daily_loss_limit, profit_target
    )

    max_days = max(n_real_days * max_days_multiplier, n_real_days)

    rng = random.Random(seed)
    outcomes = {"pass": 0, "fail_drawdown": 0, "fail_daily_loss": 0, "undetermined": 0}
    for _ in range(n_sims):
        sampled = rng.choices(daily_pnls, k=max_days)
        # Only replay as many days as needed for a resolution — cheap early
        # exit, and identical result to replaying the full max_days list
        # since _run_single_path already returns at first resolution.
        outcome = _run_single_path(
            sampled, starting_balance, drawdown_limit, daily_loss_limit, profit_target
        )
        outcomes[outcome] += 1

    stats: dict[str, Any] = {
        "n_real_days": n_real_days,
        "n_sims": n_sims,
        "max_days_simulated": max_days,
        "starting_balance": round(starting_balance, 2),
        "drawdown_limit": round(drawdown_limit, 2),
        "daily_loss_limit": round(daily_loss_limit, 2) if daily_loss_limit else None,
        "profit_target": round(profit_target, 2) if profit_target else None,
        "prob_pass": round(outcomes["pass"] / n_sims * 100, 2),
        "prob_fail_drawdown": round(outcomes["fail_drawdown"] / n_sims * 100, 2),
        "prob_fail_daily_loss": round(outcomes["fail_daily_loss"] / n_sims * 100, 2),
        "prob_undetermined": round(outcomes["undetermined"] / n_sims * 100, 2),
        "actual_outcome": actual_outcome,
    }

    return {"stats": stats, "daily_pnls": daily_pnls}
