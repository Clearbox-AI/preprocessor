from __future__ import annotations

import polars as pl


def round_half_away(expr: pl.Expr) -> pl.Expr:
    """
    Round to the nearest integer, halves away from zero.

    Polars' ``round()`` rounds halves away from zero up to 1.28 and to even from 1.29,
    so it is not used: decoded values must not depend on the installed Polars.
    """
    return expr.sign() * (expr.abs() + 0.5).floor()
