from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from elasticsearch import Elasticsearch
from openai import OpenAI
from datetime import datetime
from pydantic import BaseModel
import pathlib
import os

from setup import (
    create_index_if_not_exists,
    MESSAGES_INDEX,
    CONVERSATIONS_INDEX,
    CONVERSATIONS_MAPPING,
    MESSAGES_MAPPING,
    USERS_INDEX,
    USERS_MAPPING,
    ES_HOST,
    ES_API_KEY,
)

# ──────────────────────────────────────────────
# Clients
# ──────────────────────────────────────────────

es             = Elasticsearch(ES_HOST, api_key=ES_API_KEY)
openai_client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str) -> list:
    """Generate embedding using OpenAI text-embedding-3-small (1536 dims)."""
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# ──────────────────────────────────────────────
# In-memory store (conversation state only — users now persisted in ES)
active_conversations: dict = {}   # conversation_id -> { from_user, to_user, started_at }

# ──────────────────────────────────────────────
# App Lifespan — runs setup on startup
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 App starting — running index setup...")
    create_index_if_not_exists(es, USERS_INDEX,         USERS_MAPPING)
    create_index_if_not_exists(es, MESSAGES_INDEX,      MESSAGES_MAPPING)
    create_index_if_not_exists(es, CONVERSATIONS_INDEX, CONVERSATIONS_MAPPING)
    print("✅ Indices ready.\n")
    yield

FRONTEND_DIR = pathlib.Path(__file__).parent.parent / "frontend"

app = FastAPI(title="ES Chat Demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_ui():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

# ──────────────────────────────────────────────
# WebSocket Connection Manager
# ──────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: dict[str, WebSocket] = {}  # username -> websocket

    async def connect(self, username: str, websocket: WebSocket):
        await websocket.accept()
        self.active[username] = websocket

    def disconnect(self, username: str):
        self.active.pop(username, None)

    async def send_to(self, username: str, message: dict):
        ws = self.active.get(username)
        if ws:
            await ws.send_json(message)

manager = ConnectionManager()

# ──────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str

class StartConversationRequest(BaseModel):
    from_user: str
    to_user:   str

# ──────────────────────────────────────────────
# Routes — Users
# ──────────────────────────────────────────────

@app.post("/register")
def register(req: RegisterRequest):
    name = req.name.strip()
    if not name:
        return {"error": "Name cannot be empty"}

    # Check if user already exists in ES
    result = es.search(
        index=USERS_INDEX,
        body={"query": {"term": {"name": name}}, "size": 1}
    )
    if result["hits"]["total"]["value"] > 0:
        return {"message": f"Welcome back, {name}!", "name": name}

    # Store new user in ES
    es.index(
        index=USERS_INDEX,
        body={"name": name, "registered_at": datetime.utcnow().isoformat()}
    )
    es.indices.refresh(index=USERS_INDEX)
    return {"message": f"Welcome, {name}!", "name": name}


@app.get("/users")
def list_users():
    result = es.search(
        index=USERS_INDEX,
        body={"query": {"match_all": {}}, "size": 100}
    )
    users_list = [
        {
            "name":   hit["_source"]["name"],
            "online": hit["_source"]["name"] in manager.active
        }
        for hit in result["hits"]["hits"]
    ]
    return {"users": users_list}

# ──────────────────────────────────────────────
# Routes — Conversation
# ──────────────────────────────────────────────

@app.post("/conversation/start")
def start_conversation(req: StartConversationRequest):
    conv_id = f"{req.from_user}__{req.to_user}__{int(datetime.utcnow().timestamp())}"
    active_conversations[conv_id] = {
        "from_user":  req.from_user,
        "to_user":    req.to_user,
        "started_at": datetime.utcnow().isoformat(),
    }
    return {"conversation_id": conv_id}


@app.post("/conversation/end/{conversation_id}")
def end_conversation(conversation_id: str):
    if conversation_id not in active_conversations:
        return {"error": "Conversation not found"}

    # Fetch all messages for this conversation (sorted oldest → newest)
    result = es.search(
        index=MESSAGES_INDEX,
        body={
            "query": {"term": {"conversation_id": conversation_id}},
            "sort":  [{"timestamp": "asc"}],
            "size":  1000,
        },
    )

    messages = [hit["_source"] for hit in result["hits"]["hits"]]

    if not messages:
        del active_conversations[conversation_id]
        return {"message": "No messages found — nothing stored", "count": 0}

    # Embed each message individually → store in `conversations` index
    for msg in messages:
        embedding = get_embedding(msg["text"])
        es.index(
            index=CONVERSATIONS_INDEX,
            body={
                "conversation_id": conversation_id,
                "from_user":       msg["from_user"],
                "to_user":         msg["to_user"],
                "text":            msg["text"],
                "timestamp":       msg["timestamp"],
                "embedding":       embedding,
            },
        )

    del active_conversations[conversation_id]
    return {"message": "Conversation ended and stored in vector index", "messages_stored": len(messages)}

# ──────────────────────────────────────────────
# Routes — Messages
# ──────────────────────────────────────────────

@app.get("/messages/{conversation_id}")
def get_messages(conversation_id: str):
    result = es.search(
        index=MESSAGES_INDEX,
        body={
            "query": {"term": {"conversation_id": conversation_id}},
            "sort":  [{"timestamp": "asc"}],
            "size":  1000,
        },
    )
    return {"messages": [hit["_source"] for hit in result["hits"]["hits"]]}

# ──────────────────────────────────────────────
# Routes — Search
# ──────────────────────────────────────────────

@app.get("/search/keyword")
def keyword_search(q: str):
    """BM25 full-text search over the `messages` index."""
    result = es.search(
        index=MESSAGES_INDEX,
        body={
            "query": {"match": {"text": q}},
            "highlight": {"fields": {"text": {}}},
            "size": 20,
        },
    )
    hits = [
        {
            "text":            hit["_source"]["text"],
            "from_user":       hit["_source"]["from_user"],
            "to_user":         hit["_source"]["to_user"],
            "timestamp":       hit["_source"]["timestamp"],
            "conversation_id": hit["_source"]["conversation_id"],
            "highlight":       hit.get("highlight", {}).get("text", []),
            "score":           hit["_score"],
        }
        for hit in result["hits"]["hits"]
    ]
    return {"results": hits, "total": result["hits"]["total"]["value"]}


@app.get("/search/semantic")
def semantic_search(q: str):
    """kNN vector search over the `conversations` index."""
    query_vector = get_embedding(q)
    result = es.search(
        index=CONVERSATIONS_INDEX,
        body={
            "knn": {
                "field":        "embedding",
                "query_vector": query_vector,
                "k":            10,
                "num_candidates": 100,
            }
        },
    )
    hits = [
        {
            "text":            hit["_source"]["text"],
            "from_user":       hit["_source"]["from_user"],
            "to_user":         hit["_source"]["to_user"],
            "timestamp":       hit["_source"]["timestamp"],
            "conversation_id": hit["_source"]["conversation_id"],
            "score":           hit["_score"],
        }
        for hit in result["hits"]["hits"]
    ]
    return {"results": hits, "total": result["hits"]["total"]["value"]}

# ──────────────────────────────────────────────
# WebSocket — Real-time messaging
# ──────────────────────────────────────────────

@app.websocket("/ws/{username}")
async def websocket_endpoint(websocket: WebSocket, username: str):
    await manager.connect(username, websocket)
    try:
        while True:
            # Expected payload: { "to": str, "text": str, "conversation_id": str }
            data = await websocket.receive_json()

            msg = {
                "conversation_id": data["conversation_id"],
                "from_user":       username,
                "to_user":         data["to"],
                "text":            data["text"],
                "timestamp":       datetime.utcnow().isoformat(),
            }

            # Store in `messages` index (no embedding yet)
            es.index(index=MESSAGES_INDEX, body=msg)

            # Deliver to recipient (if online)
            await manager.send_to(data["to"], msg)

            # Echo back to sender
            await manager.send_to(username, msg)

    except WebSocketDisconnect:
        manager.disconnect(username)
