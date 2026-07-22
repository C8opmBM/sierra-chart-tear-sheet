"""Normalise raw rows into Fills events."""

from __future__ import annotations

import pandas as pd


def extract_fills(df: pd.DataFrame) -> pd.DataFrame:
    """Return only *Fills* rows, deduplicated by (InternalOrderID, FillExecutionServiceID).

    Rows without a FillExecutionServiceID (empty string) are kept as-is
    because some brokers omit that field.

    FillExecutionServiceID alone is NOT a safe dedup key: Sierra Chart can
    assign the *same* FillExecutionServiceID to multiple genuinely separate
    child-order fills that execute within the same instant as part of one
    bracket/cluster (e.g. a multi-lot auto-trade entry split across several
    child orders of different sizes, all filled essentially simultaneously).
    Deduping on FillExecutionServiceID alone silently drops those distinct
    fills, undercounting quantity and truncating downstream FIFO P&L/fee
    calculations. InternalOrderID reliably distinguishes each child order,
    so the composite key only removes rows that are true re-logged
    duplicates of the *same* order's fill.
    """
    fills = df[df["ActivityType"] == "Fills"].copy()

    if fills.empty:
        return fills

    has_id = fills["FillExecutionServiceID"].str.strip().ne("")
    with_id = fills[has_id].drop_duplicates(subset=["InternalOrderID", "FillExecutionServiceID"])
    without_id = fills[~has_id]

    result = pd.concat([with_id, without_id]).sort_values("DateTime").reset_index(drop=True)
    return result
