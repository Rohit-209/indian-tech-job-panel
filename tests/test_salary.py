"""Salary parser cases: real panel phrasings plus the Indian formats the parser claims."""

import pytest

from extraction.salary import parse_salary


@pytest.mark.parametrize("text,lo,hi,cur,monthly", [
    ("India Annual Pay Range ₹1,370,000 — ₹1,790,000 INR", 1_370_000, 1_790_000, "INR", False),
    ("CTC: INR 12,00,000 - 18,00,000 per annum", 1_200_000, 1_800_000, "INR", False),
    ("Compensation 12-18 LPA", 1_200_000, 1_800_000, "INR", False),
    ("Stipend ₹40,000 - ₹50,000 per month", 480_000, 600_000, "INR", True),
    ("The base salary range is $150,000 — $200,000 USD", 150_000, 200_000, "USD", False),
])
def test_ranges(text, lo, hi, cur, monthly):
    s = parse_salary(text)
    assert (s.min, s.max, s.currency, s.from_monthly) == (lo, hi, cur, monthly)
    assert s.period == "annual"


def test_single_value_needs_pay_context():
    s = parse_salary("Salary: up to 25 LPA for the right candidate")
    assert (s.min, s.max, s.currency) == (None, 2_500_000, "INR")
    assert parse_salary("We have 15 lakh customers").max is None


@pytest.mark.parametrize("text", [
    "5-7 years of experience in Python",
    "Work from the office 3-5 days a week",
    "Job Post Range (United States) $0.00 - $0.00 USD",
    "Levels L2 - L4 are eligible",
    "",
])
def test_non_salary_text_gives_nothing(text):
    assert parse_salary(text).max is None
