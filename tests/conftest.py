import pytest
import sys
import os
import types

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "transcription",
        types.SimpleNamespace(
            transcribe_timestamped=lambda audio, offset: [],
            SAMPLE_RATE=16000,
        ),
    )

    from server import app

    app.config.update(
        {
            "TESTING": True,
        }
    )

    # other setup can go here

    yield app

    # clean up / reset resources here


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def runner(app):
    return app.test_cli_runner()
