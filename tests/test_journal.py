import main
from conftest import register_user, login_user, auth_headers


def _logged_in_headers(client, email="journal@example.com", username="journaluser"):
    register_user(client, email=email, username=username)
    token = login_user(client, email=email).json()["access_token"]
    return auth_headers(token)


def test_create_entry_requires_auth(client):
    r = client.post("/journal/entries", json={"raw_text": "hello"})
    assert r.status_code == 401


def test_create_list_delete_entry(client, monkeypatch):
    monkeypatch.setattr(main, "analyze_with_claude", lambda text: {
        "themes": ["gratitude", "work"],
        "distress_score": 3,
        "reflection": "Sounds like a good day.",
    })
    headers = _logged_in_headers(client)

    r = client.post("/journal/entries", json={"raw_text": "Had a great day at work!"},
                     headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["llm_result"]["distress_score"] == 3
    assert body["sentiment"]["label"] in ("POSITIVE", "NEGATIVE")
    entry_id = body["entry"]["id"]

    r = client.get("/journal/entries", headers=headers)
    assert r.status_code == 200
    assert len(r.json()) == 1
    assert r.json()[0]["id"] == entry_id

    r = client.delete(f"/journal/entries/{entry_id}", headers=headers)
    assert r.status_code == 200
    assert client.get("/journal/entries", headers=headers).json() == []


def test_delete_entry_not_owned_returns_404(client, monkeypatch):
    monkeypatch.setattr(main, "analyze_with_claude", lambda text: {
        "themes": [], "distress_score": 0, "reflection": "ok"
    })
    headers_a = _logged_in_headers(client, "owner@example.com", "owneruser")
    headers_b = _logged_in_headers(client, "other@example.com", "otheruser")

    r = client.post("/journal/entries", json={"raw_text": "private thoughts"},
                     headers=headers_a)
    entry_id = r.json()["entry"]["id"]

    r = client.delete(f"/journal/entries/{entry_id}", headers=headers_b)
    assert r.status_code == 404


def test_analytics_empty_state(client):
    headers = _logged_in_headers(client, "empty@example.com", "emptyuser")
    r = client.get("/journal/analytics", headers=headers)
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_analytics_with_entries(client, monkeypatch):
    monkeypatch.setattr(main, "analyze_with_claude", lambda text: {
        "themes": [], "distress_score": 2, "reflection": "ok"
    })
    headers = _logged_in_headers(client, "stats@example.com", "statsuser")
    client.post("/journal/entries", json={"raw_text": "feeling okay"}, headers=headers)

    r = client.get("/journal/analytics", headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 1
    assert data["avg_distress"] == 2
