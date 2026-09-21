"""Which model configurations fail fast at boot, and which are let through."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import model_preflight  # noqa: E402
import startup_errors  # noqa: E402
from model_preflight import check_model_access  # noqa: E402


def hub_that_raises(error, gated=False):
    """A patched HfApi whose model_info raises `error` (or passes if None)."""
    api = MagicMock()
    if error is not None:
        api.model_info.side_effect = error
    else:
        api.model_info.return_value = SimpleNamespace(gated=gated)
    return patch.object(model_preflight, "HfApi", return_value=api), api


def hub_error(cls, message: str, status: int = 404):
    """Build a huggingface_hub error (they require a response since hub 1.x)."""
    response = SimpleNamespace(status_code=status, headers={}, request=None)
    return cls(message, response=response)


def http_error(status: int) -> HfHubHTTPError:
    return hub_error(HfHubHTTPError, f"HTTP {status}", status)


class TestDefinitiveFailures:
    def test_nonexistent_repo_fails_fast_with_the_not_found_message(self):
        patcher, _ = hub_that_raises(hub_error(RepositoryNotFoundError, "404 Client Error"))
        with patcher:
            message = check_model_access({"MODEL_NAME": "definitely-not-a-real-org/nope-123"})

        assert "definitely-not-a-real-org/nope-123 was not found" in message
        assert "MODEL_NAME" in message

    def test_gated_model_without_a_token_names_hf_token(self):
        # Gated repos serve model_info publicly; the failure surfaces on the
        # follow-up auth_check, exactly as on the real Hub.
        patcher, api = hub_that_raises(None, gated="manual")
        api.auth_check.side_effect = hub_error(GatedRepoError, "401 Client Error", 401)
        with patcher:
            message = check_model_access({"MODEL_NAME": "meta-llama/Llama-3.1-8B-Instruct"})

        assert "gated or private" in message
        assert "HF_TOKEN" in message

    def test_gated_model_with_an_accepted_token_proceeds(self):
        patcher, api = hub_that_raises(None, gated="manual")
        with patcher:
            assert check_model_access({"MODEL_NAME": "org/gated", "HF_TOKEN": "hf_ok"}) is None
        api.auth_check.assert_called_once_with("org/gated")

    def test_bad_revision_gets_its_own_message(self):
        patcher, _ = hub_that_raises(hub_error(RevisionNotFoundError, "404 Client Error"))
        with patcher:
            message = check_model_access({"MODEL_NAME": "org/model", "MODEL_REVISION": "v9"})

        assert "MODEL_REVISION=v9" in message
        assert "org/model" in message

    @pytest.mark.parametrize("status", [401, 403])
    def test_plain_auth_rejections_read_as_gated(self, status):
        patcher, _ = hub_that_raises(http_error(status))
        with patcher:
            assert "HF_TOKEN" in check_model_access({"MODEL_NAME": "org/model"})

    def test_wording_matches_the_post_crash_classifier(self):
        # The user must read the same sentence whether the failure is caught
        # here in seconds or by startup_errors.classify after a crash.
        patcher, api = hub_that_raises(None, gated="auto")
        api.auth_check.side_effect = hub_error(GatedRepoError, "401", 401)
        with patcher:
            fast = check_model_access({"MODEL_NAME": "org/model"})
        gated_crash = "huggingface_hub.errors.GatedRepoError: 401 Client Error."

        assert fast == startup_errors.classify(gated_crash, model="org/model")


class TestAmbiguityLetsBootProceed:
    @pytest.mark.parametrize(
        "error",
        [
            ConnectionError("Connection reset by peer"),
            TimeoutError("timed out"),
            http_error(500),
            http_error(503),
            http_error(429),
            OSError("Temporary failure in name resolution"),
        ],
    )
    def test_transient_errors_are_never_reported_as_not_found(self, error):
        # A network blip must get a retry (vLLM makes its own attempt), not a
        # fatal "model not found" the user cannot act on.
        patcher, _ = hub_that_raises(error)
        with patcher:
            assert check_model_access({"MODEL_NAME": "org/model"}) is None


class TestValidModel:
    def test_fetchable_model_passes_and_boot_proceeds(self):
        patcher, api = hub_that_raises(None)
        with patcher:
            assert check_model_access({"MODEL_NAME": "Qwen/Qwen2.5-0.5B-Instruct"}) is None
        api.model_info.assert_called_once()
        api.auth_check.assert_not_called()  # not gated: one metadata call is enough

    def test_checks_the_same_values_vllm_will_use(self):
        # args_builder maps MODEL_NAME -> --model, MODEL_REVISION -> --revision,
        # and vLLM reads HF_TOKEN; the pre-flight must validate those, no others.
        patcher, api = hub_that_raises(None)
        with patcher as hf_api:
            check_model_access(
                {"MODEL_NAME": "org/model", "MODEL_REVISION": "v2", "HF_TOKEN": "hf_secret"}
            )

        hf_api.assert_called_once_with(token="hf_secret")
        args, kwargs = api.model_info.call_args
        assert args == ("org/model",)
        assert kwargs["revision"] == "v2"


class TestSkippedConfigurations:
    @pytest.mark.parametrize(
        "env",
        [
            {},  # no MODEL_NAME: config-file / MODEL deploys resolve elsewhere
            {"MODEL_NAME": "org/model", "HF_HUB_OFFLINE": "1"},
            {"MODEL_NAME": "org/model", "TRANSFORMERS_OFFLINE": "1"},
            {"MODEL_NAME": "s3://bucket/model"},  # non-HF source
            {"MODEL_NAME": str(Path(__file__).parent)},  # local path
            {"MODEL_NAME": "", "VLLM_CONFIG_FILE": "/config.yaml"},
        ],
    )
    def test_never_touches_the_hub(self, env):
        patcher, api = hub_that_raises(None)
        with patcher:
            assert check_model_access(env) is None
        api.model_info.assert_not_called()
