from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.encoders import jsonable_encoder
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel, ConfigDict, EmailStr
from typing import Optional, List
from datetime import datetime

from database import get_db, init_db, User, Entry, MayaMemory, MayaInsight, ConversationLog
from auth import (hash_password, verify_password, create_access_token, get_current_user,
                  hash_security_answer, verify_security_answer, normalize_security_answer,
                  create_reset_token, verify_reset_token, SECURITY_QUESTIONS)
from llm import (chat_with_claude, analyze_with_claude,
                 summarize_conversation, extract_quick_memory)
from sentiment import analyze_sentiment

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Myndara API", version="1.0.0", lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = [
    "https://mindtrack-frontend-eight.vercel.app",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://mindtrack-frontend.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ───────────────────────────────────────────────
class UserCreate(BaseModel):
    email:             EmailStr
    username:          str
    password:          str
    security_question: str
    security_answer:   str

class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id:       int
    email:    str
    username: str

class Token(BaseModel):
    access_token: str
    token_type:   str

class RecoverStartRequest(BaseModel):
    email: EmailStr

class RecoverStartResponse(BaseModel):
    security_question: str

class RecoverVerifyRequest(BaseModel):
    email:           EmailStr
    security_answer: str

class RecoverVerifyResponse(BaseModel):
    username:    str
    reset_token: str

class RecoverResetRequest(BaseModel):
    reset_token: str
    new_password: str

class EntryCreate(BaseModel):
    raw_text: str

class EntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id:              int
    date:            str
    raw_text:        str
    sentiment_label: Optional[str]
    sentiment_score: Optional[float]
    distress_score:  Optional[int]
    themes:          Optional[str]
    llm_reflection:  Optional[str]

class ChatMessage(BaseModel):
    message: str
    history: List[dict] = []

class SaveSessionRequest(BaseModel):
    conversation: List[dict]

# ── Auth routes ────────────────────────────────────────────────────
@app.get("/auth/security-questions")
def get_security_questions():
    return {"questions": SECURITY_QUESTIONS}

@app.post("/auth/register", response_model=UserResponse)
@limiter.limit("10/hour")
def register(request: Request, user_data: UserCreate, db: Session = Depends(get_db)):
    if user_data.security_question not in SECURITY_QUESTIONS:
        raise HTTPException(status_code=400, detail="Invalid security question")
    if len(user_data.security_answer.strip()) < 2:
        raise HTTPException(status_code=400, detail="Security answer is too short")
    if len(user_data.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hash_password(user_data.password),
        security_question=user_data.security_question,
        security_answer_hash=hash_security_answer(user_data.security_answer)
    )
    db.add(user); db.commit(); db.refresh(user)
    return user

@app.post("/auth/login", response_model=Token)
@limiter.limit("10/minute")
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(),
          db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user.email})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/auth/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# ── Account recovery (security question based) ─────────────────────
_GENERIC_QUESTION = SECURITY_QUESTIONS[0]

@app.post("/auth/recover/start", response_model=RecoverStartResponse)
@limiter.limit("10/hour")
def recover_start(request: Request, body: RecoverStartRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    # Always return a question, real or generic, so this endpoint can't be used
    # to enumerate which emails have accounts.
    return {"security_question": user.security_question if user else _GENERIC_QUESTION}

@app.post("/auth/recover/verify", response_model=RecoverVerifyResponse)
@limiter.limit("10/hour")
def recover_verify(request: Request, body: RecoverVerifyRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_security_answer(body.security_answer, user.security_answer_hash):
        raise HTTPException(status_code=400, detail="Incorrect answer")
    return {"username": user.username, "reset_token": create_reset_token(user.email)}

@app.post("/auth/recover/reset")
@limiter.limit("10/hour")
def recover_reset(request: Request, body: RecoverResetRequest, db: Session = Depends(get_db)):
    if len(body.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    email = verify_reset_token(body.reset_token)
    if not email:
        raise HTTPException(status_code=400, detail="Reset link expired or invalid — please start over")
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=400, detail="Reset link expired or invalid — please start over")
    user.hashed_password = hash_password(body.new_password)
    db.commit()
    return {"status": "password updated"}

# ── Journal routes ─────────────────────────────────────────────────
@app.post("/journal/entries")
@limiter.limit("20/hour")
def create_entry(request: Request, entry_data: EntryCreate,
                 current_user: User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    sentiment  = analyze_sentiment(entry_data.raw_text)
    llm_result = analyze_with_claude(entry_data.raw_text)
    entry = Entry(
        user_id         = current_user.id,
        date            = datetime.now().strftime("%Y-%m-%d"),
        raw_text        = entry_data.raw_text,
        sentiment_label = sentiment["label"],
        sentiment_score = sentiment["mood_score"],
        distress_score  = llm_result.get("distress_score"),
        themes          = ",".join(llm_result.get("themes", [])),
        llm_reflection  = llm_result.get("reflection")
    )
    db.add(entry); db.commit(); db.refresh(entry)
    return {
        "entry":      jsonable_encoder(entry),
        "sentiment":  sentiment,
        "llm_result": llm_result
    }

@app.get("/journal/entries", response_model=List[EntryResponse])
def get_entries(current_user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    return db.query(Entry).filter(
        Entry.user_id == current_user.id
    ).order_by(Entry.created_at.desc()).all()

@app.delete("/journal/entries/{entry_id}")
def delete_entry(entry_id: int,
                 current_user: User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    entry = db.query(Entry).filter(
        Entry.id == entry_id,
        Entry.user_id == current_user.id
    ).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    db.delete(entry)
    db.commit()
    return {"status": "deleted"}

@app.get("/journal/analytics")
def get_analytics(current_user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    entries = db.query(Entry).filter(
        Entry.user_id == current_user.id
    ).order_by(Entry.created_at.asc()).all()
    if not entries:
        return {"total": 0, "avg_mood": 0, "avg_distress": 0, "best_mood": 0}
    moods      = [e.sentiment_score for e in entries if e.sentiment_score is not None]
    distresses = [e.distress_score  for e in entries if e.distress_score is not None]
    return {
        "total":        len(entries),
        "avg_mood":     round(sum(moods) / len(moods), 1) if moods else 0,
        "avg_distress": round(sum(distresses) / len(distresses), 1) if distresses else 0,
        "best_mood":    max(moods) if moods else 0,
        "trend":        [{"date": e.date, "mood": e.sentiment_score,
                          "distress": e.distress_score} for e in entries[-7:]]
    }

# ── Maya chat routes ───────────────────────────────────────────────
@app.post("/maya/chat")
@limiter.limit("60/hour")
def maya_chat(request: Request, chat_data: ChatMessage,
              current_user: User = Depends(get_current_user),
              db: Session = Depends(get_db)):

    memories = db.query(MayaMemory).filter(
        MayaMemory.user_id == current_user.id
    ).order_by(MayaMemory.created_at.desc()).limit(7).all()

    insights = db.query(MayaInsight).filter(
        MayaInsight.user_id == current_user.id
    ).order_by(MayaInsight.created_at.desc()).limit(15).all()

    today = datetime.now().strftime("%Y-%m-%d")
    conv_log = db.query(ConversationLog).filter(
        ConversationLog.user_id == current_user.id
    ).order_by(ConversationLog.created_at.asc()).all()

    days_of_data = db.query(func.count(func.distinct(MayaMemory.date)))\
                     .filter(MayaMemory.user_id == current_user.id).scalar()

    memories_list = [{"date": m.date, "summary": m.summary,
                      "dominant_emotion": m.dominant_emotion,
                      "key_topics": m.key_topics,
                      "distress_level": m.distress_level,
                      "positive_triggers": m.positive_triggers,
                      "negative_triggers": m.negative_triggers}
                     for m in memories]

    insights_list = [{"insight": i.insight, "category": i.category,
                      "created_at": str(i.created_at)} for i in insights]

    conv_list = [{"role": c.role, "content": c.content,
                  "date": str(c.created_at)} for c in conv_log[-40:]]

    reply = chat_with_claude(
        chat_data.message,
        chat_data.history,
        memories=memories_list,
        insights=insights_list,
        days_of_data=days_of_data,
        conversation_log=conv_list
    )

    # Save messages
    db.add(ConversationLog(user_id=current_user.id, session_date=today,
                           role="user", content=chat_data.message))
    db.add(ConversationLog(user_id=current_user.id, session_date=today,
                           role="assistant", content=reply))
    db.commit()

    # Quick memory
    try:
        quick = extract_quick_memory(chat_data.message, reply)
        if quick and quick.get("insight"):
            exists = db.query(MayaInsight).filter(
                MayaInsight.user_id == current_user.id,
                MayaInsight.insight == quick["insight"]
            ).first()
            if not exists:
                db.add(MayaInsight(user_id=current_user.id,
                                   insight=quick["insight"],
                                   category=quick.get("category","general")))
                db.commit()
    except Exception:
        pass

    return {"reply": reply, "days_of_data": days_of_data}

@app.post("/maya/save-session")
def save_session(session_data: SaveSessionRequest,
                 current_user: User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    if len(session_data.conversation) < 2:
        return {"status": "too short to summarize"}
    summary_data = summarize_conversation(session_data.conversation)
    if summary_data:
        db.add(MayaMemory(
            user_id          = current_user.id,
            date             = datetime.now().strftime("%Y-%m-%d"),
            summary          = summary_data.get("summary",""),
            dominant_emotion = summary_data.get("dominant_emotion",""),
            key_topics       = summary_data.get("key_topics",""),
            distress_level   = summary_data.get("distress_level",""),
            positive_triggers = summary_data.get("positive_triggers",""),
            negative_triggers = summary_data.get("negative_triggers","")
        ))
        for ins in summary_data.get("insights",[]):
            exists = db.query(MayaInsight).filter(
                MayaInsight.user_id == current_user.id,
                MayaInsight.insight == ins.get("insight","")
            ).first()
            if not exists:
                db.add(MayaInsight(
                    user_id  = current_user.id,
                    insight  = ins.get("insight",""),
                    category = ins.get("category","general")
                ))
        db.commit()
    return {"status": "saved", "summary": summary_data}

@app.get("/maya/sessions")
def get_sessions(current_user: User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    logs = db.query(ConversationLog).filter(
        ConversationLog.user_id == current_user.id
    ).order_by(ConversationLog.created_at.asc()).all()

    sessions: dict = {}
    for log in logs:
        d = log.session_date
        if d not in sessions:
            sessions[d] = {
                "id":           d,
                "created_at":   log.created_at.isoformat(),
                "conversation": []
            }
        sessions[d]["conversation"].append({
            "role":    log.role,
            "content": log.content
        })

    # Return newest sessions first
    return list(reversed(list(sessions.values())))

@app.delete("/maya/sessions/{session_date}")
def delete_session(session_date: str,
                   current_user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    db.query(ConversationLog).filter(
        ConversationLog.user_id == current_user.id,
        ConversationLog.session_date == session_date
    ).delete(synchronize_session=False)

    db.query(MayaMemory).filter(
        MayaMemory.user_id == current_user.id,
        MayaMemory.date == session_date
    ).delete(synchronize_session=False)

    db.commit()
    return {"status": "deleted"}

@app.get("/maya/memory")
def get_memory(current_user: User = Depends(get_current_user),
               db: Session = Depends(get_db)):
    memories = db.query(MayaMemory).filter(
        MayaMemory.user_id == current_user.id
    ).order_by(MayaMemory.created_at.desc()).limit(7).all()

    insights = db.query(MayaInsight).filter(
        MayaInsight.user_id == current_user.id
    ).order_by(MayaInsight.created_at.desc()).limit(15).all()

    days = db.query(func.count(func.distinct(MayaMemory.date)))\
              .filter(MayaMemory.user_id == current_user.id).scalar()

    return {
        "days_of_data": days,
        "memories": [{"date": m.date, "summary": m.summary,
                      "dominant_emotion": m.dominant_emotion,
                      "key_topics": m.key_topics,
                      "distress_level": m.distress_level}
                     for m in memories],
        "insights": [{"insight": i.insight, "category": i.category}
                     for i in insights]
    }

@app.get("/health")
def health():
    return {"status": "ok", "service": "Myndara API"}