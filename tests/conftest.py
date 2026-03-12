import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture()
def app():
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
