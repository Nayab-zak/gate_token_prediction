import polars as pl
from app.data.leakage_guards import assert_no_future_leakage, LeakageError

def test_lag_has_nulls():
    df = pl.DataFrame({"TokenCount":[1,2,3], "lag_1h":[None,1,2]})
    assert_no_future_leakage(df, "TokenCount", ["lag_1h"])

def test_lag_no_nulls_is_error():
    df = pl.DataFrame({"TokenCount":[1,2,3], "lag_1h":[1,2,3]})
    try:
        assert_no_future_leakage(df, "TokenCount", ["lag_1h"])
        assert False, "leakage not detected"
    except LeakageError:
        assert True
