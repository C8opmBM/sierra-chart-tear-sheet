"""Tests for:

* Updated broker commission rates in ``normalize/cash_ledger.py``.
* Monte Carlo simulation consuming net (post-commission) P&L instead of
  gross P&L, so simulated ruin probability / balance distribution reflect
  real trading costs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tearsheet.normalize.cash_ledger import COMMISSION_PER_SIDE, compute_fee_events_from_fills
from tearsheet.metrics.montecarlo import run_monte_carlo


class TestUpdatedCommissionRates:
    def test_rates_match_broker_confirmed_table(self):
        assert COMMISSION_PER_SIDE == {
            "MES": 0.50,
            "MNQ": 0.50,
            "ES": 1.75,
            "NQ": 1.75,
            "MCL": 0.50,
            "MGC": 0.80,
        }

    @pytest.mark.parametrize(
        "symbol,expected_rate",
        [
            ("MESU6.CME", 0.50),
            ("MNQZ6.CME", 0.50),
            ("ESU6.CME", 1.75),
            ("NQZ6.CME", 1.75),
            ("MCLQ6.CME", 0.50),
            ("MGCQ6.CME", 0.80),
        ],
    )
    def test_fee_events_use_updated_rate_per_fill(self, symbol, expected_rate):
        fills = pd.DataFrame(
            {
                "Symbol": [symbol],
                "Quantity": [2],
                "DateTime": [pd.Timestamp("2026-06-26 10:41:07")],
            }
        )
        events = compute_fee_events_from_fills(fills)
        assert len(events) == 1
        assert events[0]["amount"] == pytest.approx(2 * expected_rate)


class TestMonteCarloUsesNetPnl:
    def test_ruin_probability_higher_with_fees_than_without(self):
        # A borderline-profitable trade sequence: fees alone should push
        # more bootstrap paths into ruin territory than the gross figures do.
        gross_pnls = [50.0, -40.0, 55.0, -45.0, 60.0, -50.0, 45.0, -40.0]
        fee_per_trade = 8.0
        net_pnls = [p - fee_per_trade for p in gross_pnls]

        starting_balance = 500.0  # small on purpose to make ruin reachable
        gross_result = run_monte_carlo(gross_pnls, starting_balance, n_sims=2000)
        net_result = run_monte_carlo(net_pnls, starting_balance, n_sims=2000)

        assert net_result["stats"]["ruin_probability"] >= gross_result["stats"]["ruin_probability"]
        assert net_result["stats"]["median_final"] < gross_result["stats"]["median_final"]

    def test_actual_curve_reflects_supplied_pnls_directly(self):
        # run_monte_carlo itself is agnostic to gross vs net — it just sums
        # whatever list it's given. The fix lives in the caller (render.py),
        # so this pins run_monte_carlo's own contract: whatever pnls list it
        # receives is what the curve accumulates, unmodified.
        pnls = [10.0, -5.0, 20.0, -8.0, 15.0]
        result = run_monte_carlo(pnls, starting_balance=1000.0, n_sims=100)
        expected_curve = [1000.0]
        bal = 1000.0
        for p in pnls:
            bal += p
            expected_curve.append(bal)
        assert result["actual_curve"] == pytest.approx(expected_curve)


class TestRenderReportFeedsNetPnlToMonteCarlo:
    def test_render_module_uses_net_pnl_not_gross_pnl(self):
        # Guards against silently reverting to gross_pnl in the pnls list
        # built for run_monte_carlo inside render_report().
        import inspect

        from tearsheet.report import render

        source = inspect.getsource(render.render_report)
        # The list feeding run_monte_carlo must be built from net_pnl.
        assert 'pnls = [t.get("net_pnl", 0.0) for t in trades]' in source
        assert 'pnls = [t.get("gross_pnl", 0.0) for t in trades]' not in source
