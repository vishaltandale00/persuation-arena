"""Guards for the lint/format tooling wiring (feat: lint-tooling).

These assert the *contract* that keeps the CI lint job advisory-only and the
configs internally consistent — not the lint findings themselves (which are
expected and allowed to change). If someone makes the lint job blocking, or
drops the ruff/eslint config, these fail loudly.
"""

import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_ruff_config_present_and_sane() -> None:
    cfg = _load_pyproject()
    ruff = cfg["tool"]["ruff"]
    # A line-length must be set so ruff does not flag every long line in the
    # existing codebase; keep it generous to match current style.
    assert isinstance(ruff["line-length"], int)
    assert ruff["line-length"] >= 100
    # The noisy stylistic codes are ignored; correctness codes are NOT.
    ignored = set(ruff["lint"]["ignore"])
    assert {"E702", "E731", "F541"} <= ignored
    for keep in ("F401", "F811", "F821", "F841"):
        assert keep not in ignored, f"{keep} must stay enabled (real-bug rule)"


def test_eslint_and_prettier_config_present() -> None:
    assert (ROOT / "eslint.config.js").is_file()
    assert (ROOT / ".prettierrc.json").is_file()
    # eslint flat config must scope to the hand-written JS dirs.
    eslint_src = (ROOT / "eslint.config.js").read_text()
    assert "api/**/*.js" in eslint_src
    assert "web/**/*.js" in eslint_src


def test_package_json_declares_lint_tooling() -> None:
    pkg = yaml.safe_load((ROOT / "package.json").read_text())  # JSON is valid YAML
    dev = pkg["devDependencies"]
    assert "eslint" in dev
    assert "prettier" in dev
    assert pkg["scripts"].get("lint")


def test_ci_lint_job_is_non_blocking() -> None:
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    jobs = ci["jobs"]
    assert "lint" in jobs, "expected a dedicated lint job"
    lint = jobs["lint"]
    # The whole job must be non-blocking so it never gates merges.
    assert lint.get("continue-on-error") is True, (
        "lint job must be continue-on-error so it stays advisory"
    )
    # The pre-existing test job must remain blocking (no accidental relaxation).
    assert jobs["test"].get("continue-on-error") in (None, False)
    # Lint job actually invokes the linters.
    steps_blob = yaml.dump(lint["steps"])
    assert "ruff check" in steps_blob
    assert "eslint" in steps_blob
