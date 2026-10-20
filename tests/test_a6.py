"""A6 concentration index and the within-employer permutation test."""

import pandas as pd
import pytest

from analysis.a6_geography import hhi, within_employer_test


def test_hhi_bounds():
    assert hhi(pd.Series(["A"] * 10)) == pytest.approx(10_000)
    assert hhi(pd.Series(["A", "B", "C", "D"])) == pytest.approx(2_500)


def test_city_fully_explained_by_employer_is_not_evidence():
    # Every employer posts in one city only, and city lines up with family
    # perfectly: the pooled table looks extreme, but shuffling within
    # employers changes nothing, so p must be 1.
    df = pd.DataFrame({
        "company": ["A"] * 10 + ["B"] * 10,
        "city": ["Bengaluru"] * 10 + ["NCR"] * 10,
        "family": ["software_eng"] * 10 + ["non_tech"] * 10,
    })
    r = within_employer_test(df, n=200)
    assert r["p_perm"] == 1.0 and r["n_informative"] == 0


def test_within_employer_difference_is_detected():
    df = pd.DataFrame({
        "company": ["A"] * 40,
        "city": ["Bengaluru"] * 20 + ["NCR"] * 20,
        "family": ["software_eng"] * 20 + ["non_tech"] * 20,
    })
    assert within_employer_test(df, n=500)["p_perm"] < 0.01
