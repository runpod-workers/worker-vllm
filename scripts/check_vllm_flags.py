#!/usr/bin/env python3
"""Compare the worker's `vllm serve` flag allowlist against the real vLLM CLI.

The allowlist in src/args_builder.py is version-sensitive: vLLM renames, deletes,
and re-categorizes flags between releases (e.g. --swap-space was deleted, and every
boolean became a `--flag/--no-flag` pair that rejects `--flag false`).

dump — run where the TARGET vllm is installed, i.e. inside the pinned base image:

    VLLM_VERSION=$(sed -n 's/^ARG VLLM_VERSION=//p' Dockerfile | head -1)
    docker run --rm -v "$PWD:/repo" --entrypoint python3 \
        "vllm/vllm-openai:${VLLM_VERSION}" \
        /repo/scripts/check_vllm_flags.py dump -o /repo/.cache/vllm_serve_flags.json

check — run anywhere (stdlib only):

    python3 scripts/check_vllm_flags.py check --snapshot .cache/vllm_serve_flags.json

Exit code is non-zero when the worker could emit a CLI this vLLM rejects (a stale
or mis-categorized flag). Brand-new upstream flags are informational unless
--strict is passed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_serve_parser():
    """Construct the real `vllm serve` parser from the installed vLLM."""
    try:
        from vllm.entrypoints.openai.cli_args import make_arg_parser
    except ImportError as e:
        raise SystemExit(
            f"cannot import the vLLM serve parser ({e}); vLLM's CLI API changed, update this script"
        )
    try:  # current location (vllm/utils/argparse_utils.py)
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:
        try:  # older re-export path
            from vllm.utils import FlexibleArgumentParser
        except ImportError as e:
            raise SystemExit(
                f"cannot import FlexibleArgumentParser ({e}); vLLM's CLI API changed, update this script"
            )
    try:
        return make_arg_parser(FlexibleArgumentParser(description="vllm serve"))
    except RuntimeError as e:
        if "Failed to infer device type" not in str(e):
            raise
    # GPU-less host (CI runner, emulated docker): platform resolution yields
    # UnspecifiedPlatform, so the DeviceConfig default factory can't infer a
    # device. Flag classification doesn't depend on the device — pin CPU and
    # rebuild the parser. (`vllm.platforms.current_platform` is settable.)
    import vllm.platforms
    from vllm.platforms.cpu import CpuPlatform

    vllm.platforms.current_platform = CpuPlatform()
    return make_arg_parser(FlexibleArgumentParser(description="vllm serve"))


def classify(parser) -> dict[str, set[str]]:
    """Split the parser's long options into bool_pair / store_true / value."""
    cats: dict[str, set[str]] = {"bool_pair": set(), "store_true": set(), "value": set()}
    for action in parser._actions:
        if isinstance(action, (argparse._HelpAction, argparse._VersionAction)):
            continue
        long_opts = [o for o in action.option_strings if o.startswith("--")]
        if not long_opts:  # positional (e.g. model_tag) or subparsers
            continue
        positive = [o for o in long_opts if not o.startswith("--no-")]
        if isinstance(action, argparse.BooleanOptionalAction) or len(positive) != len(long_opts):
            cats["bool_pair"].update(positive)
        elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse._CountAction)):
            cats["store_true"].update(positive)
        else:
            cats["value"].update(positive)
    return cats


def dump(output: str | None) -> None:
    import vllm

    cats = classify(build_serve_parser())
    payload = {"vllm_version": vllm.__version__, **{k: sorted(v) for k, v in cats.items()}}
    if output:
        Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        counts = ", ".join(f"{len(v)} {k.replace('_', '-')}" for k, v in cats.items())
        print(f"wrote {output} (vllm {payload['vllm_version']}): {counts}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


def check(snapshot: str | None, strict: bool) -> int:
    if snapshot:
        data = json.loads(Path(snapshot).read_text())
        source = f"snapshot {snapshot} (vllm {data.get('vllm_version', '?')})"
        cats = {k: set(data[k]) for k in ("bool_pair", "store_true", "value")}
    else:
        try:
            import vllm
        except ImportError:
            sys.exit(
                "no --snapshot given and vllm is not importable here; "
                "generate a snapshot inside the base image (see module docstring)"
            )
        cats = classify(build_serve_parser())
        source = f"locally installed vllm {vllm.__version__}"

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from args_builder import BOOL_FLAGS, PAIRED_BOOL_FLAGS, REMOVED_FLAGS, VALUE_FLAGS

    worker = {"value": VALUE_FLAGS, "bool_pair": PAIRED_BOOL_FLAGS, "store_true": BOOL_FLAGS}
    upstream_cat = {flag: cat for cat, flags in cats.items() for flag in flags}

    errors, infos = [], []
    for worker_cat, flags in worker.items():
        for flag in sorted(flags):
            if flag not in upstream_cat:
                if flag not in REMOVED_FLAGS:
                    errors.append(
                        f"{flag}: mapped by worker ({worker_cat}) but vLLM has no such flag "
                        "— move it to REMOVED_FLAGS"
                    )
            elif upstream_cat[flag] != worker_cat:
                errors.append(f"{flag}: worker treats it as {worker_cat}, upstream it is {upstream_cat[flag]}")
    for flag in sorted(REMOVED_FLAGS):
        if flag in upstream_cat:
            errors.append(
                f"{flag}: listed in REMOVED_FLAGS but vLLM still accepts it ({upstream_cat[flag]}) "
                "— categorize it properly instead"
            )
    # Owned by the wrapper or deliberately not exposed; never reported as
    # "new upstream flag".
    WRAPPER_OWNED = {
        "--host", "--port",  # pinned by main.py
        "--config",          # VLLM_CONFIG_FILE alias
        "--api-key",         # unsupported by design: RunPod authenticates callers
    }
    worker_all = set().union(*worker.values(), set(REMOVED_FLAGS))
    for flag in sorted(set(upstream_cat) - worker_all - WRAPPER_OWNED):
        infos.append(f"{flag}: new upstream flag ({upstream_cat[flag]}), not mapped")

    print(f"checked worker allowlist against {source}")
    for msg in errors:
        print(f"  ERROR {msg}")
    for msg in infos:
        print(f"  info  {msg}")
    if errors:
        print(f"\n{len(errors)} drift error(s): the worker can emit a CLI this vLLM rejects")
    elif infos and strict:
        print(f"\n{len(infos)} new upstream flag(s) and --strict is set")
    elif not infos:
        print("  no drift")
    return 1 if errors or (strict and bool(infos)) else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="dump the installed vLLM's serve flags (run inside the base image)")
    d.add_argument("-o", "--output", help="write JSON here instead of stdout")
    c = sub.add_parser("check", help="diff the worker allowlist against a dump (stdlib only)")
    c.add_argument("--snapshot", help="previously dumped JSON; if omitted, introspect any locally installed vllm")
    c.add_argument("--strict", action="store_true", help="also fail on new, unmapped upstream flags")
    args = parser.parse_args()

    if args.cmd == "dump":
        dump(args.output)
    else:
        sys.exit(check(args.snapshot, args.strict))


if __name__ == "__main__":
    main()
