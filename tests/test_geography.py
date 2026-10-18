"""Regression tests for city canonicalisation and workplace-model precedence."""

import pytest

from extraction.geography import extract_geography, parse_location


@pytest.mark.parametrize("location,cities", [
    ("Bengaluru, India", ["Bengaluru"]),
    ("Bangalore", ["Bengaluru"]),
    ("IN-Bengaluru", ["Bengaluru"]),
    ("Gurugram", ["NCR"]),
    ("New Delhi, India", ["NCR"]),
    ("Noida, Uttar Pradesh", ["NCR"]),
    ("Bengaluru; Gurugram; Mumbai", ["Bengaluru", "NCR", "Mumbai"]),
    ("Gurugram; New Delhi", ["NCR"]),  # NCR collapse dedupes
    ("Hyderabad, Telangana, India", ["Hyderabad"]),
    ("Remote - India", []),
    ("India", []),
])
def test_cities(location, cities):
    assert parse_location(location)[0] == cities


def test_multi_country_only_counts_india_segments_for_remote():
    cities, remote, multi = parse_location(
        "Bangalore, India; Remote, Canada; Remote, United States")
    assert cities == ["Bengaluru"] and multi
    assert not remote  # the remote segments aren't the India one
    _, remote, multi = parse_location("Remote, India; Remote, Phillipines")
    assert remote and multi


def test_ats_field_beats_everything():
    g = extract_geography("ashby", "Delhi, India", "#LI-Remote",
                          {"workplaceType": "Hybrid", "isRemote": True,
                           "secondaryLocations": [{"location": "India - Remote"}]})
    assert (g.workplace, g.workplace_source) == ("hybrid", "ats_field")
    assert g.is_hybrid is True and g.is_remote is False


def test_mongodb_working_model_field():
    payload = {"metadata": [{"name": "Working Model Eligibility", "value": ["2-Flexible"]}]}
    assert extract_geography("greenhouse", "Gurugram", "", payload).workplace == "hybrid"
    payload = {"metadata": [{"name": "Working Model Eligibility",
                             "value": ["3CF-Customer Facing Remote"]}]}
    assert extract_geography("greenhouse", "Gurugram", "", payload).workplace == "remote"


def test_li_tag_then_location():
    g = extract_geography("greenhouse", "Remote - India", "Computer Science #LI-Hybrid P24182")
    assert (g.workplace, g.workplace_source) == ("hybrid", "li_tag")
    g = extract_geography("greenhouse", "Remote - India", "Build things.")
    assert (g.workplace, g.workplace_source) == ("remote", "location")


@pytest.mark.parametrize("text,level", [
    ("We use a hybrid model of 3 days in office.", "hybrid"),
    ("This is a hybrid role based out of our Bangalore office.", "hybrid"),
    ("willing to work in the office at least 2 days per week", "hybrid"),
    ("supporting an all-remote, globally distributed team", "remote"),
    ("our remote-first, asynchronous organization", "remote"),
])
def test_explicit_phrases(text, level):
    g = extract_geography("greenhouse", "Bengaluru", text)
    assert (g.workplace, g.workplace_source) == (level, "phrase")


@pytest.mark.parametrize("text", [
    "Experience working in remote or globally distributed teams",
    "develop hybrid analyzers that combine rule-based detection with AI",
    "travel up to 30% to customer sites for on-site engagements",
    "you can work from the office, or home, regardless of where you live.",
])
def test_incidental_words_give_no_signal(text):
    g = extract_geography("greenhouse", "Bengaluru", text)
    assert g.workplace is None and g.is_remote is None and g.is_hybrid is None


def test_company_policy_is_last_resort_and_labelled():
    g = extract_geography("greenhouse", "Bengaluru", "Build things.", company="Gitlab")
    assert (g.workplace, g.workplace_source) == ("remote", "company_policy")
    g = extract_geography("greenhouse", "Bengaluru", "Build things.", company="Okta")
    assert g.workplace is None
