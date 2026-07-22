"""Tests for tearsheet.normalize.fills.extract_fills — specifically the
dedup-key fix for the "shared FillExecutionServiceID across genuinely
separate child-order fills" bug.

Real-world trigger (confirmed against an actual Sierra Chart log): a
multi-lot auto-trade entry gets split across several child orders (distinct
InternalOrderID, different quantities) that all fill within the same
instant. Sierra Chart can assign the *same* FillExecutionServiceID to more
than one of those distinct fills. Deduping on FillExecutionServiceID alone
silently dropped one of the genuinely separate fills, undercounting entry
quantity and truncating the downstream FIFO gross P&L / fee calculation by
exactly the dropped fill's share.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tearsheet.normalize.fills import extract_fills


def _fill_row(dt, internal_order_id, qty, exec_id, symbol="MESU6.CME", bs="Buy"):
    return {
        "ActivityType": "Fills",
        "DateTime": pd.Timestamp(dt),
        "Symbol": symbol,
        "InternalOrderID": internal_order_id,
        "Quantity": qty,
        "BuySell": bs,
        "FillPrice": 7540.50,
        "PositionQuantity": qty,
        "OpenClose": "Open",
        "OrderType": "Market",
        "FillExecutionServiceID": exec_id,
    }


class TestExtractFillsDedup:
    def test_distinct_orders_sharing_exec_id_are_both_kept(self):
        # The exact real-world scenario: two different child orders
        # (86427 qty=1, 86430 qty=2) both stamped with FillExecutionServiceID
        # "1054290". Both are genuine, separate fills and must survive.
        df = pd.DataFrame([
            _fill_row("2026-07-20 09:49:30.410047", "86427", 1, "1054290"),
            _fill_row("2026-07-20 09:49:30.410258", "86430", 2, "1054290"),
            _fill_row("2026-07-20 09:49:30.410384", "86433", 1, "1054800"),
        ])
        result = extract_fills(df)
        assert len(result) == 3
        assert result["Quantity"].sum() == 4
        assert set(result["InternalOrderID"]) == {"86427", "86430", "86433"}

    def test_true_duplicate_same_order_same_exec_id_is_deduped(self):
        # A genuinely re-logged duplicate: identical InternalOrderID AND
        # identical FillExecutionServiceID -> should collapse to one row.
        df = pd.DataFrame([
            _fill_row("2026-07-20 09:49:30.410047", "86427", 1, "1054290"),
            _fill_row("2026-07-20 09:49:30.410047", "86427", 1, "1054290"),
        ])
        result = extract_fills(df)
        assert len(result) == 1
        assert result["Quantity"].sum() == 1

    def test_rows_without_exec_id_are_kept_as_is(self):
        df = pd.DataFrame([
            _fill_row("2026-07-20 09:49:30.410047", "86427", 1, ""),
            _fill_row("2026-07-20 09:49:30.410258", "86430", 2, ""),
        ])
        result = extract_fills(df)
        assert len(result) == 2
        assert result["Quantity"].sum() == 3

    def test_non_fills_activity_type_rows_are_excluded(self):
        df = pd.DataFrame([
            _fill_row("2026-07-20 09:49:30.410047", "86427", 1, "1054290"),
            {**_fill_row("2026-07-20 09:49:30.410258", "86430", 2, "1054290"),
             "ActivityType": "Orders"},
        ])
        result = extract_fills(df)
        assert len(result) == 1

    def test_empty_input_returns_empty(self):
        df = pd.DataFrame(columns=[
            "ActivityType", "DateTime", "Symbol", "InternalOrderID", "Quantity",
            "BuySell", "FillPrice", "PositionQuantity", "OpenClose", "OrderType",
            "FillExecutionServiceID",
        ])
        result = extract_fills(df)
        assert result.empty


class TestFullPipelineRegressionForSharedExecId:
    """End-to-end: the exact real trade this bug was found on, reconstructed
    via reconstruct_trades(), should show the true 4-contract round trip —
    not the truncated 2-contract version the bug produced.
    """

    def test_shared_exec_id_cluster_reconstructs_full_quantity_trade(self):
        from tearsheet.normalize.cash_ledger import compute_fee_events_from_fills
        from tearsheet.recon.trades import reconstruct_trades

        def row(dt, oid, qty, exec_id, pos_qty, bs="Buy", price=7540.50,
                oc="Open", order_type="Market"):
            r = _fill_row(dt, oid, qty, exec_id, bs=bs)
            r["FillPrice"] = price
            r["OpenClose"] = oc
            r["OrderType"] = order_type
            r["PositionQuantity"] = pos_qty
            return r

        # Exact values observed in the real log: entries build position
        # 0->1->3->4; exits reduce 4->3->1->0 (last exit's PositionQuantity
        # is blank/NaN in the real data, inferred via the delta fallback).
        rows = [
            row("2026-07-20 09:49:30.410047", "86427", 1, "1054290", pos_qty=1),
            row("2026-07-20 09:49:30.410258", "86430", 2, "1054290", pos_qty=3),
            row("2026-07-20 09:49:30.410384", "86433", 1, "1054800", pos_qty=4),
            row("2026-07-20 09:52:00.365198", "86435", 1, "1057068", pos_qty=3,
                bs="Sell", price=7535.25, oc="Close", order_type="Stop"),
            row("2026-07-20 09:52:00.365321", "86432", 2, "1055725", pos_qty=1,
                bs="Sell", price=7535.25, oc="Close", order_type="Stop"),
            row("2026-07-20 09:52:00.365373", "86429", 1, "1055393", pos_qty=0,
                bs="Sell", price=7535.25, oc="Close", order_type="Stop"),
        ]
        df = pd.DataFrame(rows)
        fills = extract_fills(df)
        cash_events = compute_fee_events_from_fills(fills)
        trades = reconstruct_trades(fills, cash_events)

        assert len(trades) == 1
        t = trades[0]
        assert t["total_qty"] == 4
        # 4 contracts, long, entry 7540.50 -> exit 7535.25, MES point value $5
        assert t["gross_pnl"] == pytest.approx((7535.25 - 7540.50) * 4 * 5.0)
        # 4 entry sides + 4 exit sides @ $0.50/side (MES) = $4.00
        assert t["fees"] == pytest.approx(4.00)
        assert t["net_pnl"] == pytest.approx(t["gross_pnl"] - 4.00)
