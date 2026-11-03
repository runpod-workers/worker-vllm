"""Build-time vLLM version guard (builder/check_vllm_version.py)."""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("check_vllm_version", ROOT / "builder" / "check_vllm_version.py")
cvv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cvv)


@pytest.mark.parametrize("tag", ["v0.32.0", "0.32.0", "v0.32.0-cu130", "v0.32.0-x86_64-cu129"])
def test_release_tag_matching_install_passes(tag):
    assert cvv.check("0.32.0", "0.32.0", tag) is None


def test_release_tag_with_other_install_fails():
    assert "names release 0.32.0" in cvv.check("0.31.0", "0.31.0", "v0.32.0")


@pytest.mark.parametrize("tag", ["nightly", "latest", "step-5-preview", "muse-spark-open",
                                 "deepseekv41-flash-0909", "cu129-nightly-77154ca6"])
def test_non_release_tags_pass_when_unchanged(tag):
    # Model-named day-0 and nightly images carry dev versions; the old check
    # (vllm.__version__ == tag) rejected every one of them.
    assert cvv.check("0.32.1.dev45+g1b2a666", "0.32.1.dev45+g1b2a666", tag) is None


@pytest.mark.parametrize("tag", ["v0.28.0", "step-5-preview"])
def test_downgrade_during_build_fails_for_any_tag(tag):
    # The vllm-bnb-plugin incident: base 0.28.0 silently replaced by 0.19.1.
    err = cvv.check("0.28.0", "0.19.1", tag)
    assert err and "changed during the build" in err


def _instruction_index(lines, pattern):
    return next(i for i, line in enumerate(lines) if re.search(pattern, line))


def test_dockerfile_records_before_and_verifies_after_every_pip_install():
    lines = (ROOT / "Dockerfile").read_text().splitlines()
    record = _instruction_index(lines, r"check_vllm_version\.py record")
    verify = _instruction_index(lines, r"check_vllm_version\.py verify")
    pip = [i for i, line in enumerate(lines) if "pip install" in line and not line.lstrip().startswith("#")]
    assert pip, "expected pip installs in the Dockerfile"
    assert record < min(pip) and verify > max(pip)
