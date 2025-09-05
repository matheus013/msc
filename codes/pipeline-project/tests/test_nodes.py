import pandas as pd
from kedro_starter.pipelines.data_clean.nodes import clean_names, enrich_total, to_report

def test_clean_names():
    df = pd.DataFrame({" A ": [1], "B": [2]})
    out = clean_names(df)
    assert list(out.columns) == ["a", "b"]

def test_enrich_total():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = enrich_total(df)
    assert "total" in out.columns and out["total"].tolist() == [4, 6]

def test_to_report():
    df = pd.DataFrame({"a": [1,2,3], "total": [10,5,7]})
    out = to_report(df)
    assert len(out) == 3
    assert out.iloc[0]["total"] == 10
