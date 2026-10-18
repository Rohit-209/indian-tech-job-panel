"""Regression tests for the skills dictionary.

Every ambiguous matcher (listed tokens, verb/noun collisions, boundary cases)
gets a positive and a negative case here. Real phrasings from the panel where
possible, so a dictionary edit that breaks one shows up before a replay.
"""

import pytest

from extraction.skills import extract_skills, load_dictionary, posting_text
from extraction.text import clean_text, find_boilerplate, segments


def skills(text: str, company: str | None = None) -> set[str]:
    return extract_skills(text, company)


def test_dictionary_loads_and_slugs_are_unique():
    d = load_dictionary()
    slugs = [s.slug for s in d.skills]
    assert len(slugs) == len(set(slugs))
    assert len(slugs) >= 150


@pytest.mark.parametrize("text,expected", [
    ("Python, Go, Terraform, and related technologies", "go"),
    ("Programming (Python/Go)", "go"),
    ("Solid scripting ability in Python, Go, or Shell", "go"),
    ("Preferred programming languages: golang, python", "go"),
    ("Experience using Python or R for data analysis", "r"),
    ("Java, Python, Ruby, C, C++, C#", "c"),
    ("Python/Java/Javascript/PHP/C or C++", "c"),
])
def test_listed_tokens_fire_in_enumerations(text, expected):
    assert expected in skills(text)


@pytest.mark.parametrize("text,absent", [
    ("Our Go-to-market team helps customers", "go"),
    ("Partner with Sales and Go to Market leaders", "go"),
    ("Go beyond the basics.", "go"),
    ("Software Engineer, Professional Services R&D", "r"),
    ("Present to C-suite stakeholders", "c"),
    ("We raised our Series C, led by top investors", "c"),
])
def test_listed_tokens_reject_non_language_uses(text, absent):
    assert absent not in skills(text)


@pytest.mark.parametrize("text,present,absent", [
    ("Strong JavaScript skills", "javascript", "java"),
    ("Experience in C++ and C#", "cpp", "c"),
    ("PostgreSQL and NoSQL stores", "postgresql", "sql"),
    ("Familiar with Node.js services", "nodejs", None),
    ("Built on Spring Boot", "spring", None),
    ("Built on Spring-Boot", "spring", None),
    ("Uses SpringBoot", "spring", None),
    ("We use .NET Core", "dotnet", None),
])
def test_word_boundaries(text, present, absent):
    found = skills(text)
    assert present in found
    if absent:
        assert absent not in found


@pytest.mark.parametrize("text,skill,should_match", [
    ("Building features with React and TypeScript", "react", True),
    ("Ability to react quickly to incidents", "react", False),
    ("Expert Excel skills (Power Query)", "excel", True),
    ("Microsoft Tools (Excel, Word, PowerPoint)", "excel", True),
    ("Excel in sharing information clearly", "excel", False),
    ("Kubernetes, Terraform, Helm, ArgoCD", "helm", True),
    ("Take the helm of a growing team", "helm", False),
    ("Expert-level proficiency in Swift/Objective-C", "swift", True),
    ("Move with swift execution", "swift", False),
    ("Data processing systems like Flink, Spark, Beam", "spark", True),
    ("Knowledge of IAM protocols (OAuth, OIDC, SAML)", "identity", True),
])
def test_case_sensitive_collisions(text, skill, should_match):
    assert (skill in skills(text)) is should_match


def test_suppress_for_own_employer():
    text = "Experience with MongoDB, Postgres and Redis"
    assert "mongodb" in skills(text, company="Okta")
    assert "mongodb" not in skills(text, company="Mongodb")
    assert {"postgresql", "redis"} <= skills(text, company="Mongodb")


def test_clean_text_handles_escaped_greenhouse_html():
    raw = "&lt;p&gt;&lt;strong&gt;Python&lt;/strong&gt;&amp;nbsp;and Go&lt;/p&gt;"
    assert clean_text(raw) == "Python and Go"


def test_segments_split_on_blocks_and_sentences():
    raw = "<p>We build things. You will help.</p><ul><li>Python</li><li>Go</li></ul>"
    assert segments(raw) == ["We build things.", "You will help.", "Python", "Go"]


def test_boilerplate_removed_only_when_shared_across_company():
    legal = "We use artificial intelligence, machine learning, or automated processes in hiring."
    docs = [("Acme", f"<p>{legal}</p><p>Write code in Java every single day here.</p>")] * 3
    docs += [("Acme", f"<p>{legal}</p><p>Train machine learning models in production.</p>")]
    bp = find_boilerplate(docs)
    assert len(bp["acme"]) >= 1

    java_text = posting_text("Backend Engineer", docs[0][1], bp["acme"])
    ml_text = posting_text("ML Engineer", docs[-1][1], bp["acme"])
    # The shared legal sentence no longer gives every posting machine_learning...
    assert "machine_learning" not in skills(java_text)
    # ...but a posting that genuinely asks for it keeps it.
    assert "machine_learning" in skills(ml_text)


def test_boilerplate_skips_small_companies():
    docs = [("Tiny", "<p>We use machine learning across our whole product line.</p>")] * 3
    assert "tiny" not in find_boilerplate(docs)
