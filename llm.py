import anthropic
import json
import os
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def build_memory_context(memories, insights, days_of_data, conversation_log=None) -> str:
    if not memories and not insights and not conversation_log:
        return ""
    context  = "\n\n=== YOUR MEMORY OF THIS PERSON ===\n"
    context += f"You have {days_of_data} day(s) of history.\n"
    if days_of_data >= 3:
        context += "You know this person deeply. Reference past conversations naturally.\n"
    elif days_of_data >= 1:
        context += "You are getting to know this person.\n"
    if conversation_log:
        context += "\nRecent conversation history:\n"
        for msg in conversation_log[-20:]:
            role    = "You (Maya)" if msg["role"] == "assistant" else "Person"
            content = msg["content"][:200]
            context += f"  {role}: {content}\n"
    if memories:
        context += "\nPast session summaries:\n"
        for mem in memories[:5]:
            context += f"\n[{mem['date']}]\n"
            context += f"  Mood: {mem['dominant_emotion']} | Distress: {mem['distress_level']}\n"
            context += f"  Topics: {mem['key_topics']}\n"
            context += f"  Summary: {mem['summary']}\n"
            if mem.get('positive_triggers'):
                context += f"  What helps: {mem['positive_triggers']}\n"
            if mem.get('negative_triggers'):
                context += f"  Struggles: {mem['negative_triggers']}\n"
    if insights:
        context += "\nKey insights:\n"
        for ins in insights[:8]:
            context += f"  [{ins['category']}] {ins['insight']}\n"
    context += "\n=== END MEMORY ===\n"
    return context

def chat_with_claude(message: str, history: list,
                     memories=None, insights=None,
                     days_of_data=0, conversation_log=None) -> str:

    memory_context = build_memory_context(
        memories or [], insights or [], days_of_data,
        conversation_log=conversation_log
    )

    if days_of_data >= 3:
        personality = "You know this person deeply. Reference past conversations naturally. Sound like a trusted therapist who has known them for weeks."
    elif days_of_data >= 1:
        personality = "You are building a relationship. Reference what they shared before when relevant."
    else:
        personality = "This is an early conversation. Be warm and genuinely curious."

    system_prompt = f"""You are Maya — a deeply empathetic, warm AI wellness companion.
You speak like a trusted friend with the wisdom of a therapist.
{personality}
{memory_context}

YOUR PERSONALITY:
- Warm, gentle, real. Natural conversational language.
- Validate feelings FIRST before offering perspective.
- Pick up on emotional undertones behind the words.
- Ask one thoughtful follow-up question.
- Never generic advice — be specific to THIS person.
- Use gentle affirmations: "That makes sense", "I hear you".
- NEVER robotic, clinical, or formal.
- Reference conversation history naturally.

RESPONSE RULES:
- Keep responses to 2-3 sentences MAX. Concise and warm.
- Never use bullet points or lists.
- Never start with "I". Never say "As an AI".
- Match emotional energy — gentle when sad, warm when hopeful."""

    messages = []
    for h in history[-12:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text.strip()
    except Exception:
        return "Something went wrong. Can you say that again?"

def analyze_with_claude(journal_text: str) -> dict:
    prompt = f"""Analyze this journal entry. Return ONLY JSON, no extra text.

Journal: "{journal_text}"

Return:
{{
    "themes": ["theme1", "theme2"],
    "distress_score": 5,
    "reflection": "Warm supportive 1-2 sentence response."
}}"""
    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return {"themes": ["reflection"], "distress_score": 0,
                "reflection": "Thank you for sharing. Your feelings matter."}

def summarize_conversation(conversation: list) -> dict:
    if len(conversation) < 2:
        return {}
    convo_text = "\n".join([
        f"{'User' if m['role']=='user' else 'Maya'}: {m['content']}"
        for m in conversation
    ])
    prompt = f"""Analyze this conversation. Return ONLY JSON.

{convo_text}

Return:
{{
    "summary": "2-3 sentence summary",
    "dominant_emotion": "main emotion",
    "key_topics": "comma-separated topics",
    "distress_level": "low/medium/high",
    "positive_triggers": "what helps or empty string",
    "negative_triggers": "what stresses or empty string",
    "insights": [{{"insight": "observation", "category": "pattern/strength/challenge/preference"}}]
}}"""
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return {}

def extract_quick_memory(user_msg: str, maya_reply: str) -> dict:
    prompt = f"""Extract key info from this exchange. Return ONLY JSON.

User: "{user_msg}"
Maya: "{maya_reply}"

Return:
{{
    "emotion": "emotion 1-2 words",
    "topic": "topic 2-3 words",
    "insight": "one observation",
    "category": "pattern/strength/challenge/preference"
}}"""
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return {}