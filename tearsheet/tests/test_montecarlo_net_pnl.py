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


class TestMonteCarloStartingBalanceNoDoubleCount:
    """Regression tests for the trade-1-counted-twice bug.

    When the equity curve comes from build_equity_curve_from_trades()
    (Account-Balance-less logs), equity_curve[0]["balance"] already bakes
    in trade 1's net_pnl. Monte Carlo's own pnls list also includes trade 1,
    so naively reusing equity_curve[0]["balance"] as its starting_balance
    double-counts trade 1.
    """

    def test_uses_explicit_mc_starting_balance_when_given(self):
        from tearsheet.report.render import _monte_carlo_starting_balance

        # Fallback-style curve: first point already includes trade 1's P&L.
        equity_curve = [
            {"DateTime": pd.Timestamp("2026-06-26 10:00:00"), "balance": 50050.5},
        ]
        result = _monte_carlo_starting_balance(equity_curve, mc_starting_balance=50000.0)
        assert result == 50000.0  # true pre-trade balance, not 50050.5

    def test_falls_back_to_equity_curve_first_point_when_none(self):
        # Real-Account-Balance-data path: behavior must stay unchanged.
        from tearsheet.report.render import _monte_carlo_starting_balance

        equity_curve = [{"DateTime": pd.Timestamp("2026-06-26 10:00:00"), "balance": 18000.0}]
        result = _monte_carlo_starting_balance(equity_curve, mc_starting_balance=None)
        assert result == 18000.0

    def test_empty_curve_and_none_yields_zero(self):
        from tearsheet.report.render import _monte_carlo_starting_balance

        assert _monte_carlo_starting_balance([], mc_starting_balance=None) == 0.0

    def test_no_double_count_end_to_end(self):
        # Simulates the exact fallback pipeline: build_equity_curve_from_trades
        # produces curve[0] = starting_balance + trade_1.net_pnl. Monte Carlo
        # must recover the *true* starting_balance, not curve[0]["balance"].
        from tearsheet.recon.equity import build_equity_curve_from_trades
        from tearsheet.report.render import _monte_carlo_starting_balance
        from tearsheet.metrics.montecarlo import run_monte_carlo

        true_starting_balance = 50000.0
        trades = [
            {"exit_time": pd.Timestamp(f"2026-06-26 10:{i:02d}:00"), "net_pnl": pnl}
            for i, pnl in enumerate([50.5, -37.0, 95.5, 20.0, -15.0], start=1)
        ]
        equity_curve = build_equity_curve_from_trades(trades, true_starting_balance)

        mc_sb = _monte_carlo_starting_balance(equity_curve, mc_starting_balance=true_starting_balance)
        pnls = [t["net_pnl"] for t in trades]
        result = run_monte_carlo(pnls, mc_sb, n_sims=50)

        expected_final = true_starting_balance + sum(pnls)
        assert result["actual_curve"][-1] == pytest.approx(expected_final)
        # The buggy version would have produced starting_balance + 2 * trade_1
        # + sum(trades 2..n), i.e. expected_final + trades[0]["net_pnl"].
        buggy_final = expected_final + trades[0]["net_pnl"]
        assert result["actual_curve"][-1] != pytest.approx(buggy_final)


class TestMonteCarloRiskCapital:
    """Tests for decoupling drawdown%/ruin from the (possibly inflated)
    account balance via the risk_capital parameter.
    """

    def test_dollar_curves_unaffected_by_risk_capital(self):
        # actual_curve/percentile_curves must be identical with or without
        # risk_capital — only the %-based drawdown/ruin figures may change.
        pnls = [50.5, -37.0, 95.5, 20.0, -15.0, 30.0, -60.0, 45.0]
        without = run_monte_carlo(pnls, starting_balance=50000.0, n_sims=200)
        withrc = run_monte_carlo(pnls, starting_balance=50000.0, n_sims=200, risk_capital=2000.0)

        assert without["actual_curve"] == withrc["actual_curve"]
        assert without["percentile_curves"] == withrc["percentile_curves"]
        assert without["stats"]["starting_balance"] == withrc["stats"]["starting_balance"]
        assert without["stats"]["median_final"] == withrc["stats"]["median_final"]
        assert without["stats"]["p5_final"] == withrc["stats"]["p5_final"]
        assert without["stats"]["p95_final"] == withrc["stats"]["p95_final"]

    def test_drawdown_pct_scales_up_with_smaller_risk_capital(self):
        # A losing streak that's a tiny % of $50,000 should read as a much
        # larger % of $2,000 risk capital.
        pnls = [-40.0, -50.0, -45.0, 60.0, 55.0, -35.0, 50.0, -60.0]
        big_base = run_monte_carlo(pnls, starting_balance=50000.0, n_sims=500)
        small_risk_capital = run_monte_carlo(
            pnls, starting_balance=50000.0, n_sims=500, risk_capital=2000.0
        )
        assert small_risk_capital["stats"]["median_max_dd_pct"] > big_base["stats"]["median_max_dd_pct"]
        assert small_risk_capital["stats"]["ruin_probability"] >= big_base["stats"]["ruin_probability"]

    def test_stats_report_risk_capital_when_set(self):
        pnls = [10.0, -5.0, 20.0, -8.0, 15.0]
        result = run_monte_carlo(pnls, starting_balance=50000.0, n_sims=50, risk_capital=2000.0)
        assert result["stats"]["risk_capital"] == 2000.0

    def test_stats_risk_capital_is_none_when_not_set(self):
        pnls = [10.0, -5.0, 20.0, -8.0, 15.0]
        result = run_monte_carlo(pnls, starting_balance=50000.0, n_sims=50)
        assert result["stats"]["risk_capital"] is None

    def test_zero_or_negative_risk_capital_falls_back_to_peak_based(self):
        pnls = [10.0, -5.0, 20.0, -8.0, 15.0]
        baseline = run_monte_carlo(pnls, starting_balance=50000.0, n_sims=50)
        zero_rc = run_monte_carlo(pnls, starting_balance=50000.0, n_sims=50, risk_capital=0.0)
        neg_rc = run_monte_carlo(pnls, starting_balance=50000.0, n_sims=50, risk_capital=-100.0)
        assert zero_rc["stats"]["median_max_dd_pct"] == baseline["stats"]["median_max_dd_pct"]
        assert neg_rc["stats"]["median_max_dd_pct"] == baseline["stats"]["median_max_dd_pct"]
        assert zero_rc["stats"]["risk_capital"] is None


