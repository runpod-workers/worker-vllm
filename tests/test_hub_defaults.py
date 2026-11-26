"""Hub template defaults must stay deliberate.

`.runpod/hub.json` is what the RunPod Hub form prefills for new endpoints. The
MAX_NUM_SEQS default of 32 keeps workers conservative on smaller GPUs while
leaving an explicitly set value authoritative (issue #366); there is no
code-level default and no MAX_CONCURRENCY coupling behind it.
"""

import json
from pathlib import Path

HUB_JSON = Path(__file__).resolve().parent.parent / ".runpod" / "hub.json"


def env_defaults() -> dict[str, dict]:
    return {
        entry["key"]: entry["input"]
        for entry in json.loads(HUB_JSON.read_text())["config"]["env"]
    }


def test_max_num_seqs_hub_default_is_32():
    assert env_defaults()["MAX_NUM_SEQS"]["default"] == 32


def test_max_num_seqs_description_points_at_max_concurrency():
    assert "MAX_CONCURRENCY" in env_defaults()["MAX_NUM_SEQS"]["description"]
