from __future__ import annotations
import hashlib
import polars as pl

def row_hash(df: pl.DataFrame, cols: list[str]) -> pl.Series:
    return df.select(pl.concat_str([pl.col(c).cast(pl.Utf8) for c in cols], separator="|").alias("j")).with_columns(
        pl.col("j").map_elements(lambda s: hashlib.sha256(s.encode()).hexdigest()).alias("hash")
    )["hash"]

def feature_hash(df: pl.DataFrame, cols: list[str]) -> str:
    m = hashlib.sha256()
    for c in cols:
        m.update(str(df[c].hash().sum()).encode())
    return m.hexdigest()
