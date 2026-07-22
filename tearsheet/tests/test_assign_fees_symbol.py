"""Regression test for FlatToFlatReconstructor.assign_fees matching by
symbol, not just timestamp overlap.

Real-world trigger (confirmed against an actual Sierra Chart log): an MES
trade and an MNQ trade were open concurrently (the MNQ trade's whole
lifetime fell inside the MES trade's window). Matching fee events by
timestamp alone attributed an MES fee to the MNQ trade purely because the
MNQ trade closed first and appeared earlier in the trades list.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tearsheet.recon.trades import FlatToFlatReconstructor, reconstruct_trades


def _fill(dt, bs, qty, price, pos_qty, symbol, oc="Open"):
    return {
        "DateTime": pd.Timestamp(dt),
        "BuySell": bs,
        "Quantity": qty,
        "FilledQuantity": qty,
        "FillPrice": price,
        "PositionQuantity": pos_qty,
        "Symbol": symbol,
        "OpenClose": oc,
        "OrderType": "Market",
        "HighDuringPosition": None,
        "LowDuringPosition": None,
        "FillExecutionServiceID": "",
    }


class TestSymbolAwareFeeAssignment:
    def test_overlapping_symbols_each_get_their_own_fee(self):
        # MES trade: open 12:25:14 -> close 12:29:54 (long, entered first).
        # MNQ trade: open 12:26:01 -> close 12:27:42, fully nested inside
        # the MES trade's window.
        fills_df = pd.DataFrame([
            _fill("2026-07-15 12:25:14", "Buy", 8, 7500.0, 8, "MESU6.CME"),
            _fill("2026-07-15 12:26:01", "Buy", 2, 29000.0, 2, "MNQU6.CME"),
            _fill("2026-07-15 12:27:42", "Sell", 2, 29010.0, None, "MNQU6.CME", oc="Close"),
            _fill("2026-07-15 12:29:54", "Sell", 8, 7510.0, None, "MESU6.CME", oc="Close"),
        ])

        # One fee event per symbol, timestamped inside BOTH windows on
        # purpose (12:27:00 falls inside both the MES window 12:25:14-
        # 12:29:54 and the MNQ window 12:26:01-12:27:42) to force the
        # matcher to actually use the symbol, not just get lucky on timing.
        cash_events = [
            {"DateTime": pd.Timestamp("2026-07-15 12:27:00"), "kind": "fee", "amount": 8.0, "symbol": "MESU6.CME"},
            {"DateTime": pd.Timestamp("2026-07-15 12:27:01"), "kind": "fee", "amount": 2.0, "symbol": "MNQU6.CME"},
        ]

        trades = reconstruct_trades(fills_df, cash_events)
        by_symbol = {t["symbol"]: t for t in trades}

        assert by_symbol["MESU6.CME"]["fees"] == pytest.approx(8.0)
        assert by_symbol["MNQU6.CME"]["fees"] == pytest.approx(2.0)

    def test_backward_compatible_when_event_has_no_symbol_key(self):
        # Events without a 'symbol' key (e.g. an older caller) should still
        # match by timestamp only, preserving prior behavior.
        fills_df = pd.DataFrame([
            _fill("2026-07-15 12:25:14", "Buy", 1, 7500.0, 1, "MESU6.CME"),
            _fill("2026-07-15 12:26:00", "Sell", 1, 7510.0, None, "MESU6.CME", oc="Close"),
        ])
        cash_events = [
            {"DateTime": pd.Timestamp("2026-07-15 12:25:30"), "kind": "fee", "amount": 1.0},
        ]
        trades = reconstruct_trades(fills_df, cash_events)
        assert trades[0]["fees"] == pytest.approx(1.0)
