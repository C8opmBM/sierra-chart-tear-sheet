"""Tests for tearsheet.metrics.eod_simulator — the prop-firm evaluation
pass/fail simulator (EOD trailing drawdown + daily loss limit)."""

from __future__ import annotations

import pandas as pd
import pytest

from tearsheet.metrics.eod_simulator import (
    daily_pnls_from_trades,
    simulate_evaluation,
    _run_single_path,
)


def _trade(day: str, net_pnl: float) -> dict:
    return {"exit_time": pd.Timestamp(f"{day} 15:00:00"), "net_pnl": net_pnl}


class TestDailyPnlsFromTrades:
    def test_aggregates_multiple_trades_per_day(self):
        trades = [
            _trade("2026-06-26", 50.0),
            _trade("2026-06-26", -20.0),
            _trade("2026-06-27", 30.0),
        ]
        assert daily_pnls_from_trades(trades) == [30.0, 30.0]

    def test_sorted_chronologically_regardless_of_input_order(self):
        trades = [_trade("2026-06-27", 10.0), _trade("2026-06-26", 5.0)]
        assert daily_pnls_from_trades(trades) == [5.0, 10.0]

    def test_ignores_trades_without_exit_time(self):
        trades = [_trade("2026-06-26", 10.0), {"exit_time": None, "net_pnl": 999.0}]
        assert daily_pnls_from_trades(trades) == [10.0]

    def test_empty_trades_yields_empty_list(self):
        assert daily_pnls_from_trades([]) == []


class TestTrailingFloorLockMechanic:
    def test_floor_trails_upward_before_lock_point(self):
        # starting=50000, dd=2000 -> lock_point=52000. Balance rises to
        # 51000 (below lock_point) -> floor trails to 51000-2000=49000.
        # A further drop past that floor should fail.
        outcome = _run_single_path(
            daily_pnls=[1000.0, -2001.0],  # peak 51000 -> floor 49000; day2 balance=48999 < 49000
            starting_balance=50000.0,
            drawdown_limit=2000.0,
            daily_loss_limit=None,
            profit_target=None,
        )
        assert outcome == "fail_drawdown"

    def test_floor_locks_at_starting_balance_once_lock_point_cleared(self):
        # Balance clears 52000 (lock_point), so floor should freeze at 50000
        # and NOT keep trailing to (peak - 2000) for any peak beyond 52000.
        daily_pnls = [3000.0, 5000.0, -7999.0]  # peak=58000, then drop to 50001
        outcome = _run_single_path(
            daily_pnls, starting_balance=50000.0, drawdown_limit=2000.0,
            daily_loss_limit=None, profit_target=None,
        )
        # 50001 is still above the locked floor of 50000 -> should NOT fail.
        assert outcome == "undetermined"

    def test_breaching_locked_floor_still_fails(self):
        daily_pnls = [3000.0, 5000.0, -8001.0]  # peak=58000, drop to 49999 < locked floor 50000
        outcome = _run_single_path(
            daily_pnls, starting_balance=50000.0, drawdown_limit=2000.0,
            daily_loss_limit=None, profit_target=None,
        )
        assert outcome == "fail_drawdown"


class TestDailyLossLimit:
    def test_single_bad_day_fails_immediately_even_with_healthy_trailing_dd(self):
        # Well within trailing DD (balance barely moves) but breaches the
        # daily loss limit on day 1.
        outcome = _run_single_path(
            daily_pnls=[-1300.0, 100.0, 100.0],
            starting_balance=50000.0, drawdown_limit=2000.0,
            daily_loss_limit=1200.0, profit_target=3000.0,
        )
        assert outcome == "fail_daily_loss"

    def test_day_within_daily_limit_does_not_fail(self):
        outcome = _run_single_path(
            daily_pnls=[-1199.0],
            starting_balance=50000.0, drawdown_limit=2000.0,
            daily_loss_limit=1200.0, profit_target=3000.0,
        )
        assert outcome == "undetermined"


class TestProfitTarget:
    def test_hitting_target_passes(self):
        outcome = _run_single_path(
            daily_pnls=[1000.0, 1000.0, 1000.0],
            starting_balance=50000.0, drawdown_limit=2000.0,
            daily_loss_limit=1200.0, profit_target=3000.0,
        )
        assert outcome == "pass"

    def test_no_target_set_never_passes(self):
        outcome = _run_single_path(
            daily_pnls=[10000.0],
            starting_balance=50000.0, drawdown_limit=2000.0,
            daily_loss_limit=None, profit_target=None,
        )
        assert outcome == "undetermined"

    def test_undetermined_when_days_exhausted_without_resolution(self):
        outcome = _run_single_path(
            daily_pnls=[10.0, -5.0, 8.0],
            starting_balance=50000.0, drawdown_limit=2000.0,
            daily_loss_limit=1200.0, profit_target=3000.0,
        )
        assert outcome == "undetermined"


class TestSimulateEvaluation:
    def test_empty_trades_returns_empty_stats(self):
        result = simulate_evaluation([], starting_balance=50000.0, drawdown_limit=2000.0)
        assert result["stats"] == {}

    def test_zero_starting_balance_or_drawdown_limit_returns_empty_stats(self):
        trades = [_trade("2026-06-26", 10.0)] * 5
        assert simulate_evaluation(trades, starting_balance=0.0, drawdown_limit=2000.0)["stats"] == {}
        assert simulate_evaluation(trades, starting_balance=50000.0, drawdown_limit=0.0)["stats"] == {}

    def test_probabilities_sum_to_one_hundred(self):
        trades = [_trade(f"2026-06-{d:02d}", pnl) for d, pnl in
                  zip(range(1, 15), [200, -150, 300, -400, 250, -100, 500, -600, 150, 300, -200, 400, -100, 250])]
        result = simulate_evaluation(
            trades, starting_balance=50000.0, drawdown_limit=2000.0,
            daily_loss_limit=1200.0, profit_target=3000.0, n_sims=500,
        )
        s = result["stats"]
        total = s["prob_pass"] + s["prob_fail_drawdown"] + s["prob_fail_daily_loss"] + s["prob_undetermined"]
        assert total == pytest.approx(100.0, abs=0.5)

    def test_actual_outcome_reflects_real_historical_sequence(self):
        # Real sequence hits the profit target cleanly with no bad days.
        trades = [_trade(f"2026-06-{d:02d}", 1000.0) for d in range(1, 4)]
        result = simulate_evaluation(
            trades, starting_balance=50000.0, drawdown_limit=2000.0,
            daily_loss_limit=1200.0, profit_target=3000.0, n_sims=100,
        )
        assert result["stats"]["actual_outcome"] == "pass"

    def test_a_clearly_reckless_strategy_shows_high_failure_probability(self):
        # Daily swings routinely exceed the daily loss limit -> most bootstrap
        # paths should fail on the daily loss limit almost immediately.
        trades = [_trade(f"2026-06-{d:02d}", pnl) for d, pnl in
                  zip(range(1, 11), [-1500, 1600, -1500, 1600, -1500, 1600, -1500, 1600, -1500, 1600])]
        result = simulate_evaluation(
            trades, starting_balance=50000.0, drawdown_limit=2000.0,
            daily_loss_limit=1200.0, profit_target=3000.0, n_sims=500,
        )
        assert result["stats"]["prob_fail_daily_loss"] > 50.0
