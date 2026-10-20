"""A5 labelling and split rules on hand-built spells."""

from datetime import date

import numpy as np
import pandas as pd

from analysis.a5_prediction import bootstrap, label_born, split

START, END = date(2026, 9, 7), date(2026, 10, 20)


def spell(key, first_seen, days_open, event=1, entry=0, died=False):
    # A2 spells: duration runs from publish, entry is first sighting minus publish.
    return {"posting_key": key, "first_seen": first_seen, "entry": entry,
            "duration": entry + days_open, "event": event, "censored_by_source_death": died}


def test_label_rules():
    s = pd.DataFrame([
        spell("on_day14", date(2026, 9, 10), 14),             # closes on the horizon: positive
        spell("day15", date(2026, 9, 10), 15),                # one day late: negative
        spell("open", date(2026, 9, 10), 30, event=0),        # still open: negative
        spell("repost", date(2026, 9, 10), 5),                # closed, but A7 says repost
        spell("board_gone", date(2026, 9, 10), 5, event=0, died=True),   # unknown: dropped
        spell("board_gone_late", date(2026, 9, 10), 20, event=0, died=True),  # survived 14 d
        spell("day1", START, 3),                              # live on panel day 1: not born
        spell("old", date(2026, 9, 10), 3, entry=40),         # first seen long after publish
        spell("too_new", date(2026, 10, 10), 3),              # < 14 days of follow-up
    ])
    out = label_born(s, START, END, repost_keys={"repost"}).set_index("posting_key").y
    assert out.to_dict() == {"on_day14": 1, "day15": 0, "open": 0, "repost": 0,
                             "board_gone_late": 0}


def test_split_is_by_date_and_company_levels_come_from_train():
    df = pd.DataFrame({
        "first_seen": [date(2026, 9, 8)] * 6 + [date(2026, 10, 1)] * 3,
        "company": ["Big"] * 5 + ["Small"] + ["Big", "Small", "New"],
        "y": [0, 1, 0, 0, 1, 0, 1, 0, 0],
    })
    train, test = split(df, date(2026, 9, 27))
    assert train.first_seen.max() < test.first_seen.min()
    assert list(test.company) == ["Big", "other", "other"]


def test_bootstrap_keeps_every_positive_count():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    p = {"a": np.linspace(0, 1, 8), "b": np.full(8, 0.25)}
    out = bootstrap(y, p, [("a", "b")], n=200)
    lo, hi = out["models"]["b"]["roc_auc"]
    assert lo == hi == 0.5  # constant scores are AUC 0.5 in every resample
    assert set(out["lift"]) == {"a - b"}
