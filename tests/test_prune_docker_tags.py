"""Which Docker Hub tags the weekly prune job is allowed to delete.

The rule that matters: only `test-*` / `dev-*` tags past an age cutoff are
candidates, so releases and `latest` can never be touched.
"""

import http.server
import json
import sys
import threading
import time
import types
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import prune_docker_tags as prune  # noqa: E402

NOW = datetime(2026, 1, 10, tzinfo=timezone.utc).timestamp()


def tag(name, updated):
    return {"name": name, "last_updated": updated}


OLD = "2025-12-01T00:00:00.000000Z"
RECENT = "2026-01-09T00:00:00.000000Z"


def iso_days_ago(days):
    """Timestamp relative to the real clock, for tests that run main() end to end.

    The unit tests below inject a fixed `now`, but a real run reads the system
    clock, so the stub fixture cannot use hard-coded dates.
    """
    return datetime.fromtimestamp(time.time() - days * 86400, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.000000Z"
    )


def names(tags):
    return [t["name"] for t in tags]


def test_deletes_old_ephemeral_tags():
    tags = [tag("test-abc123", OLD), tag("dev-my-branch", OLD)]
    assert names(prune.select_stale(tags, now=NOW)) == ["test-abc123", "dev-my-branch"]


def test_keeps_recent_ephemeral_tags():
    tags = [tag("test-abc123", RECENT), tag("dev-my-branch", RECENT)]
    assert prune.select_stale(tags, now=NOW) == []


@pytest.mark.parametrize("name", ["latest", "v2.8.0", "2.8.0", "main"])
def test_never_deletes_release_or_latest(name):
    # Old, but no allowed prefix matches — protected by construction.
    assert prune.select_stale([tag(name, OLD)], now=NOW) == []


def test_cutoff_is_inclusive():
    cutoff = NOW - 14 * 86400
    at_cutoff = datetime.fromtimestamp(cutoff, timezone.utc).isoformat().replace("+00:00", "Z")
    just_inside = datetime.fromtimestamp(cutoff + 1, timezone.utc).isoformat().replace("+00:00", "Z")
    assert names(prune.select_stale([tag("test-a", at_cutoff)], now=NOW)) == ["test-a"]
    assert prune.select_stale([tag("test-a", just_inside)], now=NOW) == []


def test_unparseable_timestamp_is_kept():
    # Better to leave a tag than to guess it is old.
    assert prune.select_stale([tag("test-a", "not-a-date")], now=NOW) == []
    assert prune.select_stale([tag("test-a", None)], now=NOW) == []


def test_custom_prefixes():
    tags = [tag("test-a", OLD), tag("dev-b", OLD), tag("ci-c", OLD)]
    assert names(prune.select_stale(tags, prefixes=("ci-",), now=NOW)) == ["ci-c"]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2025-12-01T00:00:00.000000Z", datetime(2025, 12, 1, tzinfo=timezone.utc).timestamp()),
        ("2025-12-01T00:00:00Z", datetime(2025, 12, 1, tzinfo=timezone.utc).timestamp()),
        ("2025-12-01T01:00:00+01:00", datetime(2025, 12, 1, tzinfo=timezone.utc).timestamp()),
        ("garbage", None),
        (None, None),
    ],
)
def test_parse_timestamp(value, expected):
    assert prune.parse_timestamp(value) == expected


def test_main_dry_run_lists_without_deleting(monkeypatch, capsys):
    fixture = [tag("test-abc123", OLD), tag("v2.8.0", OLD)]
    monkeypatch.setattr(prune, "list_tags", lambda repo, token=None: iter(fixture))
    monkeypatch.setattr(prune, "delete_tag", lambda *a, **k: pytest.fail("dry run deleted a tag"))
    monkeypatch.delenv("DOCKERHUB_USERNAME", raising=False)
    monkeypatch.delenv("DOCKERHUB_TOKEN", raising=False)

    assert prune.main(["--repo", "runpod/worker-v1-vllm", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "test-abc123" in out
    assert "v2.8.0" not in out


def test_main_requires_credentials_for_real_run(monkeypatch):
    monkeypatch.delenv("DOCKERHUB_USERNAME", raising=False)
    monkeypatch.delenv("DOCKERHUB_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        prune.main(["--repo", "runpod/worker-v1-vllm"])


# --- destructive path, exercised against a stub Docker Hub -------------------


def test_validate_prefixes_rejects_empty():
    assert prune.validate_prefixes(["test-"]) == ("test-",)
    assert prune.validate_prefixes([]) == prune.DEFAULT_PREFIXES
    # An empty prefix would match every tag, releases included.
    with pytest.raises(SystemExit):
        prune.validate_prefixes([""])


def test_main_refuses_empty_prefix_before_touching_the_network(monkeypatch):
    monkeypatch.setattr(prune, "list_tags", lambda *a, **k: pytest.fail("listed tags"))
    monkeypatch.setattr(prune, "login", lambda *a, **k: pytest.fail("logged in"))
    with pytest.raises(SystemExit):
        prune.main(["--repo", "runpod/worker-v1-vllm", "--prefix", "", "--dry-run"])


@pytest.fixture
def stub_hub():
    """A local stand-in for the Docker Hub API that records real HTTP traffic."""
    state = {"deleted": [], "auth": [], "logins": [], "fail_delete": set(), "base_url": None}
    tags = [
        tag("test-aaa", iso_days_ago(30)),
        tag("dev-x", iso_days_ago(30)),
        tag("test-bbb", iso_days_ago(30)),
        tag("v2.8.0", iso_days_ago(30)),
        tag("latest", iso_days_ago(30)),
        tag("2.8.0", iso_days_ago(30)),
        tag("test-ccc", iso_days_ago(1)),
    ]

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, payload=None):
            body = json.dumps(payload).encode() if payload is not None else b""
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            state["logins"].append(json.loads(self.rfile.read(length) or b"{}"))
            self._send(200, {"token": "stub-token"})

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if not parsed.path.endswith("/tags/"):
                self._send(404, {"detail": "not found"})
                return
            page = int(urllib.parse.parse_qs(parsed.query).get("page", ["1"])[0])
            page_size = 2  # deliberately smaller than the request, to force paging
            start = (page - 1) * page_size
            chunk = tags[start : start + page_size]
            nxt = f"{parsed.path}?page={page + 1}" if start + page_size < len(tags) else None
            self._send(200, {"count": len(tags), "next": nxt, "results": chunk})

        def do_DELETE(self):
            name = urllib.parse.unquote(self.path.rstrip("/").rsplit("/", 1)[-1])
            state["auth"].append(self.headers.get("Authorization"))
            if name in state["fail_delete"]:
                self._send(500, {"detail": "boom"})
                return
            state["deleted"].append(name)
            self._send(204)

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    state["base_url"] = f"http://127.0.0.1:{httpd.server_address[1]}/v2"
    try:
        yield types.SimpleNamespace(**state)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _run_against_stub(monkeypatch, stub, *extra):
    monkeypatch.setenv("DOCKERHUB_API_BASE", stub.base_url)
    monkeypatch.setenv("DOCKERHUB_USERNAME", "user")
    monkeypatch.setenv("DOCKERHUB_TOKEN", "pat")
    monkeypatch.setattr(prune, "DELETE_PAUSE_SECONDS", 0)
    return prune.main(["--repo", "runpod/worker-v1-vllm", *extra])


def test_real_run_deletes_only_stale_ephemeral_tags(stub_hub, monkeypatch):
    assert _run_against_stub(monkeypatch, stub_hub) == 0
    # Paginated listing still reaches every page; only old test-/dev- tags go.
    assert sorted(stub_hub.deleted) == ["dev-x", "test-aaa", "test-bbb"]
    assert "latest" not in stub_hub.deleted
    assert "v2.8.0" not in stub_hub.deleted
    assert "2.8.0" not in stub_hub.deleted
    assert "test-ccc" not in stub_hub.deleted


def test_real_run_sends_bearer_token_on_deletes(stub_hub, monkeypatch):
    _run_against_stub(monkeypatch, stub_hub)
    assert stub_hub.logins == [{"username": "user", "password": "pat"}]
    assert stub_hub.auth and all(auth == "Bearer stub-token" for auth in stub_hub.auth)


def test_delete_failure_aborts_with_nonzero_exit(stub_hub, monkeypatch):
    stub_hub.fail_delete.add("test-bbb")
    with pytest.raises(SystemExit):
        _run_against_stub(monkeypatch, stub_hub)


def test_dry_run_against_stub_deletes_nothing(stub_hub, monkeypatch):
    monkeypatch.setenv("DOCKERHUB_API_BASE", stub_hub.base_url)
    monkeypatch.delenv("DOCKERHUB_USERNAME", raising=False)
    monkeypatch.delenv("DOCKERHUB_TOKEN", raising=False)
    assert prune.main(["--repo", "runpod/worker-v1-vllm", "--dry-run"]) == 0
    assert stub_hub.deleted == []
    assert stub_hub.logins == []
