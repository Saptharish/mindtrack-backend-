import main
from conftest import register_user, login_user, auth_headers


def _logged_in_headers(client, email="maya@example.com", username="mayauser"):
    register_user(client, email=email, username=username)
    token = login_user(client, email=email).json()["access_token"]
    return auth_headers(token)


def test_chat_requires_auth(client):
    r = client.post("/maya/chat", json={"message": "hi", "history": []})
    assert r.status_code == 401


def test_chat_returns_reply_and_saves_log(client, monkeypatch):
    monkeypatch.setattr(main, "chat_with_claude", lambda *a, **k: "I'm here for you.")
    monkeypatch.setattr(main, "extract_quick_memory", lambda *a, **k: {})
    headers = _logged_in_headers(client)

    r = client.post("/maya/chat", json={"message": "I had a hard day", "history": []},
                     headers=headers)
    assert r.status_code == 200, r.text
    assert r.json()["reply"] == "I'm here for you."

    r = client.get("/maya/sessions", headers=headers)
    assert r.status_code == 200
    sessions = r.json()
    assert len(sessions) == 1
    assert len(sessions[0]["conversation"]) == 2


def test_memory_endpoint_defaults(client):
    headers = _logged_in_headers(client, "memory@example.com", "memoryuser")
    r = client.get("/maya/memory", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["days_of_data"] == 0
    assert body["memories"] == []
    assert body["insights"] == []


def test_save_session_too_short(client):
    headers = _logged_in_headers(client, "short@example.com", "shortsession")
    r = client.post("/maya/save-session", json={"conversation": [{"role": "user", "content": "hi"}]},
                     headers=headers)
    assert r.status_code == 200
    assert r.json()["status"] == "too short to summarize"
