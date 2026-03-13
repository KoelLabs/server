import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import request_integrity


def test_protect_request_allows_verified_android(app, monkeypatch):
    monkeypatch.setattr(
        request_integrity,
        "_verify_play_integrity",
        lambda **kwargs: True,
    )

    with app.test_request_context(
        "/analyze_file",
        headers={
            "X-App-Integrity-Status": "verified",
            "X-App-Integrity-Platform": "android",
            "X-App-Integrity-Request-Hash": "abc12345",
            "X-App-Integrity-Token": "token",
        },
    ):
        assert (
            request_integrity.protect_request(
                route_key="analyze_file", expected_request_hash="abc12345"
            )
            is None
        )


def test_protect_request_rejects_invalid_verified_android(app, monkeypatch):
    monkeypatch.setattr(
        request_integrity,
        "_verify_play_integrity",
        lambda **kwargs: False,
    )

    with app.test_request_context(
        "/analyze_file",
        headers={
            "X-App-Integrity-Status": "verified",
            "X-App-Integrity-Platform": "android",
            "X-App-Integrity-Request-Hash": "abc12345",
            "X-App-Integrity-Token": "token",
        },
    ):
        response, status_code = request_integrity.protect_request(
            route_key="analyze_file", expected_request_hash="abc12345"
        )  # type: ignore
        assert status_code == 403
        assert response.get_json() == {"error": "Request integrity verification failed"}


def test_protect_request_rate_limits_fallback_requests(app, monkeypatch):
    monkeypatch.setattr(
        request_integrity, "_consume_rate_limit", lambda route_key: False
    )

    with app.test_request_context("/feedback_report"):
        response, status_code = request_integrity.protect_request(
            route_key="feedback_report", expected_request_hash="ignored"
        )  # type: ignore
        assert status_code == 429
        assert response.get_json() == {
            "error": "Too many requests. Please try again later."
        }
