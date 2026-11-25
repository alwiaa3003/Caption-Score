import pandas as pd

from src.export_colab import stratified_by_month


def test_every_month_keeps_its_share():
    train = pd.DataFrame({"month": ["a"] * 1000 + ["b"] * 3000, "x": range(4000)})
    sub = stratified_by_month(train, 400)
    counts = sub["month"].value_counts().to_dict()
    assert counts == {"a": 100, "b": 300}


def test_is_deterministic_and_never_oversamples():
    train = pd.DataFrame({"month": ["a"] * 50, "x": range(50)})
    assert stratified_by_month(train, 20).equals(stratified_by_month(train, 20))
    assert len(stratified_by_month(train, 10_000)) == 50
