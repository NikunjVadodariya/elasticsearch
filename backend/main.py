from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from elasticsearch import Elasticsearch
from openai import OpenAI
from datetime import datetime
from pydantic import BaseModel
import asyncio
import pathlib
import os

from setup import (
    create_index_if_not_exists,
    MESSAGES_INDEX,
    ensure_message_fields,
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

# ──────────────────────────────────────────────
# App Lifespan — runs setup on startup
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 App starting — running index setup...")
    create_index_if_not_exists(es, USERS_INDEX,         USERS_MAPPING)
    create_index_if_not_exists(es, MESSAGES_INDEX,      MESSAGES_MAPPING)
    # Older `messages` indexes have no embedding field yet — add it
    ensure_message_fields(es)
    print("✅ Indices ready.\n")
    yield

FRONTEND_DIR = pathlib.Path(__file__).parent / "frontend"

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
# Routes — Messages
# ──────────────────────────────────────────────

@app.get("/messages")
def get_messages(user: str, partner: str):
    """All messages between two users, oldest first."""
    result = es.search(
        index=MESSAGES_INDEX,
        body={
            "query": {"bool": {"should": [
                {"bool": {"filter": [{"term": {"from_user": user}},    {"term": {"to_user": partner}}]}},
                {"bool": {"filter": [{"term": {"from_user": partner}}, {"term": {"to_user": user}}]}},
            ], "minimum_should_match": 1}},
            "sort":  [{"timestamp": "asc"}],
            "size":  1000,
            "_source": {"excludes": ["embedding", "text_sayt"]},
        },
    )
    return {"messages": [hit["_source"] for hit in result["hits"]["hits"]]}

# ──────────────────────────────────────────────
# Routes — Search
# ──────────────────────────────────────────────

def pair_filter(user: str | None, partner: str | None) -> list:
    """Restrict a search to the chat between two users (both directions)."""
    if not user or not partner:
        return []
    return [{"bool": {"should": [
        {"bool": {"filter": [{"term": {"from_user": user}},    {"term": {"to_user": partner}}]}},
        {"bool": {"filter": [{"term": {"from_user": partner}}, {"term": {"to_user": user}}]}},
    ], "minimum_should_match": 1}}]


@app.get("/search/keyword")
def keyword_search(q: str, user: str | None = None, partner: str | None = None):
    """BM25 full-text search over the `messages` index."""
    result = es.search(
        index=MESSAGES_INDEX,
        body={
            "query": {"bool": {"must": {"match": {"text": q}}, "filter": pair_filter(user, partner)}},
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
            "highlight":       hit.get("highlight", {}).get("text", []),
            "score":           hit["_score"],
        }
        for hit in result["hits"]["hits"]
    ]
    return {"results": hits, "total": result["hits"]["total"]["value"]}


@app.get("/search/autocomplete")
def autocomplete_search(q: str, user: str | None = None, partner: str | None = None):
    """Search-as-you-type: matches partial words ("pas" -> "password")."""
    result = es.search(
        index=MESSAGES_INDEX,
        body={
            "_source": {"excludes": ["embedding", "text_sayt"]},
            "size": 10,
            "query": {"bool": {
                "must": {"multi_match": {
                    "query":  q,
                    "type":   "bool_prefix",
                    "fields": ["text_sayt", "text_sayt._2gram", "text_sayt._3gram"],
                }},
                "filter": pair_filter(user, partner),
            }},
            "highlight": {"fields": {"text_sayt": {}}, "pre_tags": ["<em>"], "post_tags": ["</em>"]},
        },
    )
    hits = [
        {
            "text":      hit["_source"]["text"],
            "from_user": hit["_source"]["from_user"],
            "to_user":   hit["_source"]["to_user"],
            "timestamp": hit["_source"]["timestamp"],
            "score":     hit["_score"],
        }
        for hit in result["hits"]["hits"]
    ]
    return {"results": hits, "total": result["hits"]["total"]["value"]}


@app.get("/search/semantic")
def semantic_search(q: str, min_similarity: float = 0.35, user: str | None = None, partner: str | None = None):
    """kNN vector search over message embeddings in `messages`."""
    query_vector = get_embedding(q)
    result = es.search(
        index=MESSAGES_INDEX,
        body={
            "_source": {"excludes": ["embedding", "text_sayt"]},
            "knn": {
                "field":        "embedding",
                "query_vector": query_vector,
                "k":            10,
                "num_candidates": 100,
                # Minimum raw cosine similarity (-1..1). Without it kNN always
                # returns the k nearest docs, even unrelated ones like "hi".
                "similarity":   min_similarity,
                "filter":       pair_filter(user, partner),
            }
        },
    )
    hits = [
        {
            "text":            hit["_source"]["text"],
            "from_user":       hit["_source"]["from_user"],
            "to_user":         hit["_source"]["to_user"],
            "timestamp":       hit["_source"]["timestamp"],
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
            # Expected payload: { "to": str, "text": str }
            data = await websocket.receive_json()

            msg = {
                "from_user":       username,
                "to_user":         data["to"],
                "text":            data["text"],
                "timestamp":       datetime.utcnow().isoformat(),
            }

            # Deliver first so chat stays instant, then embed + store
            await manager.send_to(data["to"], msg)
            await manager.send_to(username, msg)

            # Store in `messages` with its embedding (one vector per message)
            doc = {**msg, "text_sayt": msg["text"], "embedding": await asyncio.to_thread(get_embedding, msg["text"])}
            await asyncio.to_thread(lambda: es.index(index=MESSAGES_INDEX, body=doc))

    except WebSocketDisconnect:
        manager.disconnect(username)
