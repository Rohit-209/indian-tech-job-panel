"""A3 statistics against published worked examples, and the pre-registration's integrity."""

import pytest

from analysis.a3_drift import bh, load_prereg, two_prop, wilson


def test_wilson_matches_newcombe_1998():
    # Newcombe (1998) Table I: 81/263 -> 0.2553 to 0.3662
    lo, hi = wilson(81, 263)
    assert lo == pytest.approx(0.2553, abs=1e-4) and hi == pytest.approx(0.3662, abs=1e-4)


def test_difference_ci_matches_newcombe_1998_method_10():
    # Newcombe (1998) example (a): 56/70 vs 48/80 -> diff 0.2, CI 0.0524 to 0.3339
    r = two_prop(48, 80, 56, 70)  # p2 - p1 = 56/70 - 48/80
    assert r["diff"] == pytest.approx(0.2)
    assert r["lo"] == pytest.approx(0.0524, abs=1e-4)
    assert r["hi"] == pytest.approx(0.3339, abs=1e-4)


def test_bh_textbook_example():
    p = [0.01, 0.04, 0.03, 0.005, 0.20]
    adj, rej = bh(p, q=0.05)
    assert list(rej) == [True, True, True, True, False]
    assert adj[3] == pytest.approx(0.025)  # smallest p: 0.005 * 5 / 1


def test_preregistration_is_complete_and_frozen_shape():
    r = load_prereg()
    assert str(r["registered_on"]) == "2026-10-18"
    assert r["periods"]["p1"]["end"] < r["periods"]["p2"]["start"]
    assert len(r["secondary"]["per_skill"]["skills"]) == 27
    assert set(r["primary"]["genai_skills"]) >= {"llm", "genai", "ai_agents", "rag"}
    assert r["secondary"]["per_skill"]["multiplicity"]["q"] == 0.05
