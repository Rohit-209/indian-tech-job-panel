"""Regression tests for the rules title classifier and evaluation helpers.

Cases are real panel titles, mostly ones the first draft of the rules got
wrong — each pins down why a rule sits where it does in the order.
"""

import pytest

from extraction.titles import (
    FAMILIES,
    SENIORITIES,
    _mcnemar_exact,
    _wilson,
    classify_rules,
    in_scope,
)
from extraction.titles_llm import RESPONSE_SCHEMA, title_key


@pytest.mark.parametrize("title,family,seniority", [
    ("Senior Software Engineer", "software_engineering", "senior"),
    ("Staff Software Engineer - Backend", "software_engineering", "staff"),
    ("Principal Software Engineer - macOS", "software_engineering", "principal"),
    ("Software Engineer, Intern", "software_engineering", "intern"),
    ("Engineering Manager, Tech Foundations", "software_engineering", "manager"),
    ("Okta Manager, Engineering", "software_engineering", "manager"),
    ("Director, Site Reliability Engineering", "sre_infra", "director"),
    ("Senior Software Engineer in Test", "qa_testing", "senior"),
    ("Staff Machine Learning Engineer (L4)", "ml_engineering", "staff"),
    ("Data Science Manager, Risk", "data_science", "manager"),
    ("Senior Data Engineer", "data_engineering", "senior"),
    ("Senior Analyst, Advanced Analytics", "data_analytics", "senior"),
    ("Senior Solutions Architect", "solutions_support", "senior"),
    ("Technical Account Manager", "solutions_support", "unspecified"),
    ("Enterprise Account Executive, Mumbai", "non_tech", "unspecified"),
])
def test_common_titles(title, family, seniority):
    assert classify_rules(title) == (family, seniority)


@pytest.mark.parametrize("title,family", [
    # product role named after a data team, not a data engineer
    ("Senior Product Manager, Data Platform", "product_design"),
    # recruiting for engineering is still recruiting
    ("Director of Recruiting, Engineering & IT, Bangalore", "non_tech"),
    # "Detection" here is anomaly detection on an SRE team
    ("Senior Site Reliability Engineer - Monitoring and Anomaly Detection", "sre_infra"),
    # an EM for a deployment team, not a deployment (solutions) role
    ("Senior Engineering Manager - Continuous Deployment", "software_engineering"),
    # support beats the bare "developer" keyword
    ("Developer Support Engineer", "solutions_support"),
    # explicit SWE title beats customer-facing team name
    ("Software Engineer, Professional Services R&D", "software_engineering"),
    ("Consulting Architect - Search", "solutions_support"),
    ("Senior InfoSec Analyst", "security_engineering"),
    ("Tech Lead Manager, Oracle ERP", "software_engineering"),
])
def test_rule_order(title, family):
    assert classify_rules(title)[0] == family


@pytest.mark.parametrize("title,seniority", [
    ("Software Engineer 3, AI Builder Experience", "senior"),
    ("Engineer 3 - Business Systems", "senior"),
    ("Software Engineer (L2)", "mid"),
    ("Staff, Software Engineer (L4)", "staff"),
    ("Technical Support Engineer 2", "mid"),
    ("Associate TSE II", "junior"),  # "Associate" outranks the ladder
    ("Intermediate Backend Engineer", "mid"),  # GitLab's mid level
    ("Sr. Accountant", "senior"),
    ("Associate Renewals Manager, India", "junior"),  # IC "manager" job name
    ("Product Manager, Product Platform", "unspecified"),
    ("Senior Manager, Marketing", "manager"),
    ("Lead - Advanced Analytics, Gurgaon", "staff"),
])
def test_seniority(title, seniority):
    assert classify_rules(title)[1] == seniority


def test_in_scope():
    assert in_scope("software_engineering", "senior")
    assert in_scope("data_analytics", "unspecified")
    assert not in_scope("software_engineering", "manager")
    assert not in_scope("software_engineering", "intern")
    assert not in_scope("solutions_support", "senior")
    assert not in_scope("non_tech", "junior")


def test_llm_schema_matches_taxonomy():
    props = RESPONSE_SCHEMA["items"]["properties"]
    assert props["title_family"]["enum"] == list(FAMILIES)
    assert props["seniority"]["enum"] == list(SENIORITIES)


def test_title_key_normalises_case_and_space():
    a = title_key("Okta", "Senior  Software Engineer ")
    assert a == title_key("okta", "senior software engineer")
    assert title_key("Okta", "Engineer 3") != title_key("Mongodb", "Engineer 3")


def test_wilson_interval():
    lo, hi = _wilson(135, 150)
    assert 0.84 < lo < 0.86 and 0.93 < hi < 0.95
    assert _wilson(0, 0) == (0.0, 0.0)


def test_mcnemar_exact():
    assert _mcnemar_exact(0, 0) == 1.0
    assert _mcnemar_exact(5, 5) == 1.0
    assert _mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
