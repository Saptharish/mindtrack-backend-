import main
from conftest import register_user, login_user, auth_headers


def test_register_login_and_me(client):
    r = register_user(client, email="alice@example.com", username="alice", answer="Whiskers")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["email"] == "alice@example.com"
    assert body["username"] == "alice"
    assert "password" not in body
    assert "security_answer" not in body

    r = login_user(client, email="alice@example.com")
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]

    r = client.get("/auth/me", headers=auth_headers(token))
    assert r.status_code == 200
    assert r.json()["username"] == "alice"


def test_register_rejects_invalid_security_question(client):
    r = register_user(client, email="bob@example.com", username="bob",
                       question="Not a real question")
    assert r.status_code == 400


def test_register_rejects_short_password(client):
    r = client.post("/auth/register", json={
        "email": "short@example.com",
        "username": "shortpw",
        "password": "123",
        "security_question": main.SECURITY_QUESTIONS[0],
        "security_answer": "answer",
    })
    assert r.status_code == 400


def test_register_rejects_duplicate_email(client):
    register_user(client, email="dup@example.com", username="dupuser1")
    r = register_user(client, email="dup@example.com", username="dupuser2")
    assert r.status_code == 400
    assert "email" in r.json()["detail"].lower()


def test_register_rejects_duplicate_username(client):
    register_user(client, email="dup1@example.com", username="dupname")
    r = register_user(client, email="dup2@example.com", username="dupname")
    assert r.status_code == 400
    assert "username" in r.json()["detail"].lower()


def test_login_rejects_wrong_password(client):
    register_user(client, email="carol@example.com", username="carol")
    r = login_user(client, email="carol@example.com", password="wrong-password")
    assert r.status_code == 401


def test_me_requires_auth(client):
    r = client.get("/auth/me")
    assert r.status_code == 401


def test_password_recovery_full_flow(client):
    register_user(client, email="dave@example.com", username="dave", answer="Sparky")

    r = client.post("/auth/recover/start", json={"email": "dave@example.com"})
    assert r.status_code == 200
    assert r.json()["security_question"] == main.SECURITY_QUESTIONS[0]

    r = client.post("/auth/recover/verify", json={
        "email": "dave@example.com", "security_answer": "sparky"  # case-insensitive
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["username"] == "dave"
    reset_token = body["reset_token"]

    r = client.post("/auth/recover/reset", json={
        "reset_token": reset_token, "new_password": "BrandNewPass1"
    })
    assert r.status_code == 200

    assert login_user(client, "dave@example.com", "BrandNewPass1").status_code == 200
    assert login_user(client, "dave@example.com", "SuperSecret1").status_code == 401


def test_recovery_wrong_answer_is_rejected(client):
    register_user(client, email="erin@example.com", username="erin", answer="Buddy")
    r = client.post("/auth/recover/verify", json={
        "email": "erin@example.com", "security_answer": "wrong-answer"
    })
    assert r.status_code == 400


def test_recovery_unknown_email_does_not_leak_existence(client):
    r = client.post("/auth/recover/start", json={"email": "nobody@example.com"})
    assert r.status_code == 200
    assert r.json()["security_question"] == main.SECURITY_QUESTIONS[0]


def test_reset_token_cannot_be_used_as_access_token(client):
    register_user(client, email="frank@example.com", username="frank", answer="Max")
    r = client.post("/auth/recover/verify", json={
        "email": "frank@example.com", "security_answer": "Max"
    })
    reset_token = r.json()["reset_token"]

    r = client.get("/auth/me", headers=auth_headers(reset_token))
    assert r.status_code == 401
