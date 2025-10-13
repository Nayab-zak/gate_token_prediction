import polars as pl
from app.utils.validation import assert_required_columns, ValidationError

def test_missing_columns():
    df = pl.DataFrame({"A":[1,2]})
    try:
        assert_required_columns(df, ["A","B"])
        assert False, "Should raise"
    except ValidationError:
        assert True
