"""Tests for the realized-P&L equity curve fallback.

Covers the case where a Sierra Chart Trade Activity Log has no ``Account
Balance`` rows at all (observed on some Rithmic Direct - DTC / prop-firm
eval-and-funded account routings), so the normal
:func:`tearsheet.recon.equity.build_equity_curve` returns an empty list.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tearsheet.recon.equity import (
    build_equity_curve,
    build_equity_curve_from_trades,
    adjust_equity_curve,
)


def _trade(exit_time: str, net_pnl: float) -> dict:
    return {"exit_time": pd.Timestamp(exit_time), "net_pnl": net_pnl}


class TestBuildEquityCurveFromTrades:
    def test_accumulates_net_pnl_over_starting_balance(self):
        trades = [
            _trade("2026-06-26 10:41:07", 50.5),
            _trade("2026-06-26 10:45:00", -37.0),
            _trade("2026-06-26 11:00:00", 95.5),
        ]
        curve = build_equity_curve_from_trades(trades, starting_balance=50000.0)

        assert [p["balance"] for p in curve] == [50050.5, 50013.5, 50109.0]
        assert curve[0]["DateTime"] == pd.Timestamp("2026-06-26 10:41:07")

    def test_defaults_to_zero_starting_balance(self):
        trades = [_trade("2026-06-26 10:41:07", 10.0)]
        curve = build_equity_curve_from_trades(trades)
        assert curve[0]["balance"] == 10.0

    def test_sorts_by_exit_time_regardless_of_input_order(self):
        trades = [
            _trade("2026-06-26 11:00:00", 10.0),
            _trade("2026-06-26 10:00:00", 5.0),
        ]
        curve = build_equity_curve_from_trades(trades, starting_balance=0.0)
        assert [p["DateTime"] for p in curve] == [
            pd.Timestamp("2026-06-26 10:00:00"),
            pd.Timestamp("2026-06-26 11:00:00"),
        ]
        assert [p["balance"] for p in curve] == [5.0, 15.0]

    def test_ignores_open_trades_without_exit_time(self):
        trades = [_trade("2026-06-26 10:41:07", 10.0), {"exit_time": None, "net_pnl": 999.0}]
        curve = build_equity_curve_from_trades(trades, starting_balance=0.0)
        assert len(curve) == 1

    def test_empty_trades_yields_empty_curve(self):
        assert build_equity_curve_from_trades([], starting_balance=1000.0) == []

    def test_adjust_equity_curve_is_a_noop_without_cash_flows(self):
        trades = [_trade("2026-06-26 10:41:07", 10.0)]
        curve = build_equity_curve_from_trades(trades, starting_balance=1000.0)
        adjust_equity_curve(curve, [])
        assert curve[0]["adjusted_balance"] == curve[0]["balance"] == 1010.0


class TestBuildEquityCurveNoAccountBalanceRows:
    def test_returns_empty_when_no_account_balance_rows_present(self):
        # Mirrors a log with only Orders/Fills/Positions rows (no
        # "Account Balance" ActivityType at all).
        df = pd.DataFrame(
            {
                "ActivityType": ["Orders", "Fills", "Positions"],
                "DateTime": pd.to_datetime(
                    ["2026-06-26 10:41:07", "2026-06-26 10:41:07", "2026-06-26 10:41:07"]
                ),
                "AccountBalance": [None, None, None],
                "OrderActionSource": ["", "", ""],
            }
        )
        assert build_equity_curve(df) == []
