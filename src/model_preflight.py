"""Fail "the model cannot be fetched" in seconds instead of after the cold start.

When `vllm serve` downloads the weights itself (no baked-in model), a typo'd
MODEL_NAME, a gated model without HF_TOKEN, or a bad MODEL_REVISION only
surfaces once the download attempt fails at the end of a ~20-minute cold
start. One metadata call to the Hugging Face Hub answers the same question in
seconds, before vLLM is even launched, so main.py asks it first and answers
jobs with the cause (never crash-looping, same rule as startup_errors.py).

Only a definitive answer from the Hub fails the boot. A timeout, a 5xx, or any
other ambiguity lets the boot proceed: vLLM makes its own download attempt and
startup_errors.classify still catches the crash, so a transient network blip
is never misreported as "model not found".
"""

import logging
import os
from typing import Mapping, Optional

from huggingface_hub import HfApi
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

import startup_errors

# One metadata request; generous enough for a slow Hub, tiny next to the
# STARTUP_TIMEOUT it protects.
PREFLIGHT_TIMEOUT = float(os.getenv("MODEL_PREFLIGHT_TIMEOUT", "15"))

TRUE_VALUES = {"true", "1", "yes", "on"}


def _is_offline(env: Mapping[str, str]) -> bool:
    """Weights are already on disk (Option 2 builds); the Hub must not be hit."""
    return any(
        str(env.get(name, "")).strip().lower() in TRUE_VALUES
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def check_model_access(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The startup_error for an unfetchable model, or None to proceed with boot.

    Validates the same values args_builder.py hands to `vllm serve`
    (MODEL_NAME, MODEL_REVISION, HF_TOKEN) with a metadata-only Hub call —
    no weights are downloaded.
    """
    env = os.environ if env is None else env

    model = (env.get("MODEL_NAME") or "").strip()
    if not model:
        # MODEL / VLLM_CONFIG_FILE deploys name the model elsewhere; let vLLM
        # resolve those itself.
        return None
    if _is_offline(env):
        return None
    if os.path.exists(model) or "://" in model:
        # A local path or a non-HF source (e.g. the Run:ai model streamer);
        # not the Hub's question to answer.
        return None

    revision = (env.get("MODEL_REVISION") or "").strip() or None
    token = (env.get("HF_TOKEN") or "").strip() or None

    try:
        api = HfApi(token=token)
        info = api.model_info(model, revision=revision, timeout=PREFLIGHT_TIMEOUT)
        if getattr(info, "gated", False):
            # Gated repos serve their metadata publicly; only a file-access
            # check answers "may this worker download the weights?".
            api.auth_check(model)
    except GatedRepoError:
        return startup_errors.gated_message(model)
    except RevisionNotFoundError:
        return startup_errors.revision_not_found_message(model, revision or "main")
    except RepositoryNotFoundError:
        return startup_errors.not_found_message(model)
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403):
            return startup_errors.gated_message(model)
        logging.warning("Model pre-flight got HTTP %s from the HF Hub; proceeding with boot: %s", status, e)
        return None
    except Exception as e:  # DNS failure, timeout, TLS trouble, ...
        logging.warning("Model pre-flight could not reach the HF Hub; proceeding with boot: %s", e)
        return None

    logging.info("Model pre-flight: %s is fetchable", model if not revision else f"{model}@{revision}")
    return None
