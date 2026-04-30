from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime
import json

from database import get_db, init_db, User, Entry, MayaMemory, MayaInsight, ConversationLog
from auth import hash_password, verify_password, create_access_token, get_current_user
from llm import (chat_with_claude, analyze_with_claude,
                 summarize_conversation, extract_quick_memory)
from sentiment import analyze_sentiment

app = FastAPI(title="MindTrack API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    init_db()

# ── Pydantic schemas ───────────────────────────────────────────────
class UserCreate(BaseModel):
    email:    EmailStr
    username: str
    password: str

class UserResponse(BaseModel):
    id:       int
    email:    str
    username: str
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type:   str

class EntryCreate(BaseModel):
    raw_text: str

class EntryResponse(BaseModel):
    id:              int
    date:            str
    raw_text:        str
    sentiment_label: Optional[str]
    sentiment_score: Optional[float]
    distress_score:  Optional[int]
    themes:          Optional[str]
    llm_reflection:  Optional[str]
    class Config:
        from_attributes = True

class ChatMessage(BaseModel):
    message: str
    history: List[dict] = []

class SaveSessionRequest(BaseModel):
    conversation: List[dict]

# ── Auth routes ────────────────────────────────────────────────────
@app.post("/auth/register", response_model=UserResponse)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hash_password(user_data.password)
    )
    db.add(user); db.commit(); db.refresh(user)
    return user

@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(),
          db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user.email})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/auth/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# ── Journal routes ─────────────────────────────────────────────────
@app.post("/journal/entries")
def create_entry(entry_data: EntryCreate,
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
        "entry":      entry,
        "sentiment":  sentiment,
        "llm_result": llm_result
    }

@app.get("/journal/entries", response_model=List[EntryResponse])
def get_entries(current_user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    return db.query(Entry).filter(
        Entry.user_id == current_user.id
    ).order_by(Entry.created_at.desc()).all()

@app.get("/journal/analytics")
def get_analytics(current_user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    entries = db.query(Entry).filter(Entry.user_id == current_user.id).all()
    if not entries:
        return {"total": 0, "avg_mood": 0, "avg_distress": 0, "best_mood": 0}
    moods      = [e.sentiment_score for e in entries if e.sentiment_score]
    distresses = [e.distress_score  for e in entries if e.distress_score]
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
def maya_chat(chat_data: ChatMessage,
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

    days_of_data = db.query(MayaMemory).filter(
        MayaMemory.user_id == current_user.id
    ).distinct(MayaMemory.date).count()

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

@app.get("/maya/memory")
def get_memory(current_user: User = Depends(get_current_user),
               db: Session = Depends(get_db)):
    memories = db.query(MayaMemory).filter(
        MayaMemory.user_id == current_user.id
    ).order_by(MayaMemory.created_at.desc()).limit(7).all()

    insights = db.query(MayaInsight).filter(
        MayaInsight.user_id == current_user.id
    ).order_by(MayaInsight.created_at.desc()).limit(15).all()

    days = db.query(MayaMemory).filter(
        MayaMemory.user_id == current_user.id
    ).distinct(MayaMemory.date).count()

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
    return {"status": "ok", "service": "MindTrack API"}