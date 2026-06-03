"""
Guardrail: measurement-reference tables must stay out of src/pyhqiv.

Measured-value comparisons belong in tests/data and tests only.
"""

from pathlib import Path


FORBIDDEN_TOKENS = (
    "nucleon_proton_ground_MeV",
    "nucleon_neutron_ground_MeV",
    "delta_1232_MeV",
    "n_1440_MeV",
    "n_1520_MeV",
    "n_1535_MeV",
    "n_1650_MeV",
    "pdg_reference_masses_mev",
    "percent_errors_vs_pdg",
    "2.725",
    "938.272",
    "0.51099895",
    "1776.86",
    "2.224575",
    "197.327",
    "6.582119569e-22",
    "1.054571817e-34",
    "2.99792458e8",
    "6.67430e-11",
    "1.602176634e-10",
    # removed the = forms; current code assigns from loaders (values live in json)
    # Bare "PDG"/"CODATA" allowed in comments/docs; value literals + table names are forbidden in src .py
)


def test_no_measurement_reference_tokens_in_src_pyhqiv():
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src" / "pyhqiv"
    offenders: list[str] = []
    for py_file in src_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for token in FORBIDDEN_TOKENS:
            if token in text:
                offenders.append(f"{py_file.relative_to(repo_root)} :: {token}")
    assert not offenders, "Forbidden measurement-reference tokens found in src:\n" + "\n".join(offenders)
