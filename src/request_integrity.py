import hashlib
import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import requests
from flask import jsonify, request

_ROOT_APP_CONFIG_PATH = Path(__file__).resolve().parents[2] / "app.json"
_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = {}
_GOOGLE_CREDENTIALS = None
_GOOGLE_CREDENTIALS_LOCK = threading.Lock()

INTEGRITY_SCOPE = "https://www.googleapis.com/auth/playintegrity"
PLAY_INTEGRITY_DECODE_URL = (
    "https://playintegrity.googleapis.com/v1/{package_name}:decodeIntegrityToken"
)

RATE_LIMITS = {
    "analyze_file": {"limit": 30, "window_seconds": 60},
    "feedback_report": {"limit": 10, "window_seconds": 300},
}


def _load_app_config() -> dict[str, Any]:
    if not _ROOT_APP_CONFIG_PATH.exists():
        return {}

    try:
        return json.loads(_ROOT_APP_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


_APP_CONFIG = _load_app_config()
_EXPECTED_ANDROID_PACKAGE_NAME = (
    _APP_CONFIG.get("expo", {}).get("android", {}).get("package")
)


def stable_stringify(value: Any) -> str:
    if value is None or isinstance(value, (bool, int, float, str)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)

    if isinstance(value, list):
        return "[" + ",".join(stable_stringify(item) for item in value) + "]"

    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: item[0])
        return (
            "{"
            + ",".join(
                f"{json.dumps(key, ensure_ascii=False)}:{stable_stringify(nested_value)}"
                for key, nested_value in items
            )
            + "}"
        )

    return json.dumps(str(value), ensure_ascii=False)


def fnv1a32(value: str) -> str:
    hash_value = 0x811C9DC5
    for byte in value.encode("utf-8"):
        hash_value ^= byte
        hash_value = (hash_value * 0x01000193) & 0xFFFFFFFF
    return f"{hash_value:08x}"


def build_analyze_file_request_hash(
    *,
    target_words: list[str],
    metadata_json: str | None,
    file_bytes: bytes,
) -> str:
    return fnv1a32(
        stable_stringify(
            {
                "fileMd5": hashlib.md5(file_bytes).hexdigest(),
                "fileSize": len(file_bytes),
                "metadataJson": metadata_json or "",
                "route": "analyze_file",
                "targetWords": target_words,
            }
        )
    )


def build_feedback_report_request_hash(
    *,
    payload_json: str,
    screenshot_bytes: bytes | None,
) -> str:
    return fnv1a32(
        stable_stringify(
            {
                "payloadJson": payload_json,
                "route": "feedback_report",
                "screenshotMd5": (
                    hashlib.md5(screenshot_bytes).hexdigest()
                    if screenshot_bytes is not None
                    else ""
                ),
                "screenshotSize": len(screenshot_bytes) if screenshot_bytes else 0,
            }
        )
    )


def _get_google_credentials():
    global _GOOGLE_CREDENTIALS

    service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not service_account_json:
        return None

    try:
        from google.auth.transport.requests import Request as GoogleAuthRequest
        from google.oauth2 import service_account
    except ImportError:
        return None

    with _GOOGLE_CREDENTIALS_LOCK:
        if _GOOGLE_CREDENTIALS is None:
            info = json.loads(service_account_json)
            _GOOGLE_CREDENTIALS = service_account.Credentials.from_service_account_info(
                info, scopes=[INTEGRITY_SCOPE]
            )

        if not _GOOGLE_CREDENTIALS.valid:
            _GOOGLE_CREDENTIALS.refresh(GoogleAuthRequest())

        return _GOOGLE_CREDENTIALS


def _get_client_ip() -> str:
    forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _consume_rate_limit(route_key: str) -> bool:
    config = RATE_LIMITS[route_key]
    now = time.time()
    bucket_key = f"{route_key}:{_get_client_ip()}"

    with _RATE_LIMIT_LOCK:
        bucket = _RATE_LIMIT_BUCKETS.setdefault(bucket_key, deque())
        while bucket and now - bucket[0] > config["window_seconds"]:
            bucket.popleft()

        if len(bucket) >= config["limit"]:
            return False

        bucket.append(now)
        return True


def _verify_play_integrity(
    *, token: str, provided_request_hash: str, expected_request_hash: str
) -> bool | None:
    credentials = _get_google_credentials()
    if credentials is None or not _EXPECTED_ANDROID_PACKAGE_NAME:
        return None

    if provided_request_hash != expected_request_hash:
        return False

    response = requests.post(
        PLAY_INTEGRITY_DECODE_URL.format(package_name=_EXPECTED_ANDROID_PACKAGE_NAME),
        headers={
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json",
        },
        json={"integrity_token": token},
        timeout=10,
    )
    if response.status_code != 200:
        return False

    payload = response.json().get("tokenPayloadExternal", {})
    request_details = payload.get("requestDetails", {})
    app_integrity = payload.get("appIntegrity", {})
    device_integrity = payload.get("deviceIntegrity", {})

    if request_details.get("requestHash") != expected_request_hash:
        return False
    if request_details.get("requestPackageName") != _EXPECTED_ANDROID_PACKAGE_NAME:
        return False
    if app_integrity.get("appRecognitionVerdict") != "PLAY_RECOGNIZED":
        return False

    device_verdicts = set(device_integrity.get("deviceRecognitionVerdict") or [])
    if not (
        "MEETS_DEVICE_INTEGRITY" in device_verdicts
        or "MEETS_STRONG_INTEGRITY" in device_verdicts
    ):
        return False

    return True


def protect_request(*, route_key: str, expected_request_hash: str):
    status = request.headers.get("X-App-Integrity-Status", "").strip().lower()

    if status == "verified":
        platform = request.headers.get("X-App-Integrity-Platform", "").strip().lower()
        request_hash = request.headers.get("X-App-Integrity-Request-Hash", "").strip()
        token = request.headers.get("X-App-Integrity-Token", "").strip()

        if platform == "android" and token and request_hash:
            verification_result = _verify_play_integrity(
                token=token,
                provided_request_hash=request_hash,
                expected_request_hash=expected_request_hash,
            )
            if verification_result is True:
                return None
            if verification_result is False:
                return (
                    jsonify({"error": "Request integrity verification failed"}),
                    403,
                )

        else:
            return (
                jsonify({"error": "Request integrity verification failed"}),
                403,
            )

    if not _consume_rate_limit(route_key):
        return jsonify({"error": "Too many requests. Please try again later."}), 429

    return None
