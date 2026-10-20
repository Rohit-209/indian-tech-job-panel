"""The release must never include the employers' text."""

from tools.export_release import DERIVED_COLUMNS, POSTING_COLUMNS, WITHHELD


def test_withheld_columns_are_not_released():
    assert not WITHHELD & set(POSTING_COLUMNS + DERIVED_COLUMNS)
