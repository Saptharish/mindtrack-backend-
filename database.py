from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mindtrack.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True, index=True)
    email         = Column(String, unique=True, index=True, nullable=False)
    username      = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow)
    entries       = relationship("Entry", back_populates="user")
    memories      = relationship("MayaMemory", back_populates="user")
    insights      = relationship("MayaInsight", back_populates="user")
    conversations = relationship("ConversationLog", back_populates="user")

class Entry(Base):
    __tablename__ = "entries"
    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False)
    date            = Column(String, nullable=False)
    raw_text        = Column(Text, nullable=False)
    sentiment_label = Column(String)
    sentiment_score = Column(Float)
    distress_score  = Column(Integer)
    themes          = Column(String)
    llm_reflection  = Column(Text)
    created_at      = Column(DateTime, default=datetime.utcnow)
    user            = relationship("User", back_populates="entries")

class MayaMemory(Base):
    __tablename__ = "maya_memory"
    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=False)
    date             = Column(String, nullable=False)
    summary          = Column(Text, nullable=False)
    dominant_emotion = Column(String)
    key_topics       = Column(String)
    distress_level   = Column(String)
    positive_triggers = Column(String)
    negative_triggers = Column(String)
    created_at       = Column(DateTime, default=datetime.utcnow)
    user             = relationship("User", back_populates="memories")

class MayaInsight(Base):
    __tablename__ = "maya_insights"
    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    insight    = Column(Text, nullable=False)
    category   = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    user       = relationship("User", back_populates="insights")

class ConversationLog(Base):
    __tablename__ = "conversation_log"
    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_date = Column(String, nullable=False)
    role         = Column(String, nullable=False)
    content      = Column(Text, nullable=False)
    created_at   = Column(DateTime, default=datetime.utcnow)
    user         = relationship("User", back_populates="conversations")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)