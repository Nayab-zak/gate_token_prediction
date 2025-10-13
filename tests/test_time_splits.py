import polars as pl
from app.data.splits import walk_forward_by_time

def test_walk_forward_ordered():
    df = pl.DataFrame({
        "MoveDate": ["2024-01-01"]*10,
        "MoveHour": list(range(10)),
        "TerminalID": ["T1"]*10,
        "Desig": ["EXP"]*10,
        "MoveType": ["In"]*10,
        "TokenCount": list(range(10)),
    })
    splits = walk_forward_by_time(df, folds=3)
    assert len(splits) == 3
    # Ensure increasing sizes
    assert splits[0][0][-1] < splits[0][1][0]
