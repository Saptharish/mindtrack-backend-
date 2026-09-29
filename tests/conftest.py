import os
import tempfile

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")

_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"

from fastapi.testclient import TestClient

import main

# Rate limiting is a production concern; disable it for tests so a full
# test run (which creates many users from the same "IP") isn't throttled.
main.limiter.enabled = False


@pytest.fixture(scope="session", autouse=True)
def _cleanup_db():
    yield
    try:
        os.remove(_db_path)
    except OSError:
        pass


@pytest.fixture()
def client():
    with TestClient(main.app) as c:
        yield c


def register_user(client, email="user@example.com", username="testuser",
                   password="SuperSecret1", question=None, answer="Rex"):
    if question is None:
        question = main.SECURITY_QUESTIONS[0]
    return client.post("/auth/register", json={
        "email": email,
        "username": username,
        "password": password,
        "security_question": question,
        "security_answer": answer,
    })


def login_user(client, email="user@example.com", password="SuperSecret1"):
    return client.post("/auth/login", data={"username": email, "password": password})


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}
