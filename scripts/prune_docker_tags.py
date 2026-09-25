#!/usr/bin/env python3
"""Delete stale ephemeral Docker Hub tags for the worker image.

CI pushes a fresh tag per run — `test-<model_slug>-<run_id>` for each PR smoke
test, `test-<sha>` for each push to main — and `dev-<branch>` per branch. None
of them are ever cleaned up, so the tag list grows without bound.

This script deletes only tags whose name starts with one of the allowed
prefixes (`test-` and `dev-` by default) and that are older than
`--min-age-days`. Releases (`v1.2.3` / `1.2.3`) and `latest` are protected by
construction: they match no prefix, so they are never candidates.

Docker Hub has no native tag-retention policy, hence the API calls below.

Dry run (listing is public, so no credentials needed):

    python scripts/prune_docker_tags.py --dry-run

Real run (deletes need a Docker Hub username + PAT):

    DOCKERHUB_USERNAME=... DOCKERHUB_TOKEN=... \\
        python scripts/prune_docker_tags.py --min-age-days 14

Exit code is non-zero only on an API/auth failure, never merely because a tag
could not be parsed or nothing was old enough.

`DOCKERHUB_API_BASE` overrides the Docker Hub API root (used by tests to point
at a stub server; defaults to https://hub.docker.com/v2).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

DEFAULT_API_BASE = "https://hub.docker.com/v2"
DEFAULT_PREFIXES = ("test-", "dev-")
PAGE_SIZE = 100
DELETE_PAUSE_SECONDS = 0.5


def log(msg: str) -> None:
    print(f"[prune_docker_tags] {msg}", flush=True)


def api_base() -> str:
    """Docker Hub API root. Overridable so tests can point at a stub server."""
    return os.environ.get("DOCKERHUB_API_BASE", DEFAULT_API_BASE).rstrip("/")


def _request(method: str, url: str, token: str | None = None, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = resp.read()
    return json.loads(payload) if payload else {}


def login(username: str, password: str) -> str:
    """Exchange username + PAT for the JWT the delete endpoint expects."""
    try:
        response = _request(
            "POST", f"{api_base()}/users/login/", body={"username": username, "password": password}
        )
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Docker Hub login failed ({e.code}): {e.read().decode(errors='replace')}")
    token = response.get("token")
    if not token:
        raise SystemExit("Docker Hub login returned no token")
    return token


def list_tags(repo: str, token: str | None = None):
    """Yield every tag object in the repository, newest page first."""
    page = 1
    while True:
        url = f"{api_base()}/repositories/{repo}/tags/?page_size={PAGE_SIZE}&page={page}"
        try:
            response = _request("GET", url, token=token)
        except urllib.error.HTTPError as e:
            raise SystemExit(f"listing tags failed ({e.code}): {e.read().decode(errors='replace')}")
        results = response.get("results", [])
        if not results:
            return
        yield from results
        if not response.get("next"):
            return
        page += 1


def delete_tag(repo: str, tag: str, token: str) -> None:
    quoted = urllib.parse.quote(tag, safe="")
    url = f"{api_base()}/repositories/{repo}/tags/{quoted}/"
    try:
        _request("DELETE", url, token=token)
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"deleting {tag} failed ({e.code}): {e.read().decode(errors='replace')}"
        )


def parse_timestamp(value: str | None) -> float | None:
    """Epoch seconds for Docker Hub's ISO-8601 `last_updated`, or None."""
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def select_stale(tags, prefixes=DEFAULT_PREFIXES, min_age_days=14, now=None):
    """Return the tags eligible for deletion.

    A tag qualifies only when its name starts with an allowed prefix AND its
    `last_updated` is at (or before) the cutoff. Tags with an unparseable or
    missing timestamp are kept — conservative by design, since guessing would
    risk deleting a release.
    """
    if now is None:
        now = time.time()
    cutoff = now - min_age_days * 86400
    stale = []
    for tag in tags:
        name = tag.get("name", "")
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        updated = parse_timestamp(tag.get("last_updated"))
        if updated is None:
            continue
        if updated <= cutoff:
            stale.append(tag)
    return stale


def validate_prefixes(prefixes) -> tuple[str, ...]:
    """Reject prefixes that would match everything.

    An empty prefix makes `name.startswith(prefix)` true for every tag — releases
    included — so it must never reach the delete loop.
    """
    cleaned = tuple(p.strip() for p in prefixes)
    if not cleaned:
        return DEFAULT_PREFIXES
    for prefix in cleaned:
        if not prefix:
            raise SystemExit("--prefix must not be empty (an empty prefix matches every tag)")
    return cleaned


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="runpod/worker-v1-vllm", help="Docker Hub repository as namespace/name")
    parser.add_argument(
        "--prefix",
        action="append",
        dest="prefixes",
        help=f"Tag prefix eligible for deletion (repeatable; default: {' '.join(DEFAULT_PREFIXES)})",
    )
    parser.add_argument(
        "--min-age-days",
        type=float,
        default=14,
        help="Only delete tags not updated in this many days (default: 14)",
    )
    parser.add_argument("--dry-run", action="store_true", help="List what would be deleted; delete nothing")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    prefixes = validate_prefixes(args.prefixes or DEFAULT_PREFIXES)
    if "/" not in args.repo:
        raise SystemExit(f"--repo must be namespace/name, got {args.repo!r}")

    token = None
    if not args.dry_run:
        username = os.environ.get("DOCKERHUB_USERNAME")
        password = os.environ.get("DOCKERHUB_TOKEN")
        if not username or not password:
            raise SystemExit("DOCKERHUB_USERNAME and DOCKERHUB_TOKEN are required unless --dry-run")
        log(f"logging in to Docker Hub as {username}")
        token = login(username, password)

    tags = list(list_tags(args.repo, token=token))
    stale = select_stale(tags, prefixes=prefixes, min_age_days=args.min_age_days)

    log(
        f"{args.repo}: {len(tags)} tags, {len(stale)} matching "
        f"{'|'.join(prefixes)} older than {args.min_age_days:g}d"
    )
    if not stale:
        return 0

    if args.dry_run:
        for tag in stale:
            log(f"  would delete {tag.get('name')} (updated {tag.get('last_updated')})")
        log("dry run: nothing deleted")
        return 0

    for tag in stale:
        name = tag.get("name")
        delete_tag(args.repo, name, token)
        log(f"  deleted {name} (updated {tag.get('last_updated')})")
        time.sleep(DELETE_PAUSE_SECONDS)
    log(f"deleted {len(stale)} tags")
    return 0


if __name__ == "__main__":
    sys.exit(main())
