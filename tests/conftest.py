import sys
import threading
import types
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import main


@pytest.fixture
def fake_sdk(monkeypatch):
    sdk = types.SimpleNamespace(hooks=[], handler=types.ModuleType("handler"))
    sdk.handler.handler = lambda job: None

    def start(config):
        for hook in sdk.hooks:
            hook()

    sdk.start = mock.MagicMock(side_effect=start)
    runpod_stub = types.ModuleType("runpod")
    runpod_stub.serverless = types.SimpleNamespace(
        register_prestart_hook=sdk.hooks.append, start=sdk.start
    )
    monkeypatch.setitem(sys.modules, "handler", sdk.handler)
    monkeypatch.setitem(sys.modules, "runpod", runpod_stub)
    monkeypatch.setattr(main, "backend_settled", threading.Event())
    monkeypatch.delenv("RUNPOD_REALTIME_PORT", raising=False)
    return sdk
