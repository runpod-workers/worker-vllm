"""Build-time guard: the base image's vLLM must survive the worker's pip installs.

Run twice from the Dockerfile:

    python3 check_vllm_version.py record              # before any pip install
    python3 check_vllm_version.py verify "$VLLM_VERSION"  # after the last one

`verify` fails the build if the installed vLLM differs from the one recorded,
which is how a dependency once silently replaced vLLM 0.28.0 with 0.19.1
(vllm-bnb-plugin, #335). When VLLM_VERSION is a release tag (v0.32.0,
v0.32.0-cu130) it also checks the installed version is that release.

Any other base tag (nightly, latest, or a model-named day-0 image such as
step-5-preview) has no release number to compare against, so only the
"unchanged during the build" check applies. Comparing such tags to
vllm.__version__ directly would reject them outright.
"""

import re
import sys
from pathlib import Path

RECORD = Path("/usr/local/share/worker-vllm/base_vllm_version")
RELEASE_TAG = re.compile(r"^v?(\d+\.\d+\.\d+)(?:[-+].*)?$")


def installed_version() -> str:
    import vllm

    return vllm.__version__


def check(base: str, installed: str, requested_tag: str) -> str | None:
    """Return an error message, or None if the installed vLLM is acceptable."""
    if installed != base:
        return (f"vLLM changed during the build: base image had {base}, now {installed}. "
                "A pip install replaced it; pin or --no-deps the offending package.")
    m = RELEASE_TAG.match(requested_tag)
    if m and installed != m.group(1):
        return (f"VLLM_VERSION={requested_tag} names release {m.group(1)}, "
                f"but the base image contains vLLM {installed}.")
    return None


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "record":
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        RECORD.write_text(installed_version())
        print(f"base image vLLM: {RECORD.read_text()}")
        return 0
    if len(argv) == 3 and argv[1] == "verify":
        installed = installed_version()
        err = check(RECORD.read_text().strip(), installed, argv[2])
        if err:
            print(f"ERROR: {err}", file=sys.stderr)
            return 1
        print(f"vLLM {installed} OK for base tag {argv[2]}")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
