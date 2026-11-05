import os
from pathlib import Path
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ──────────────────────────────────────────────
# Constants — index names used across the app
# ──────────────────────────────────────────────
MESSAGES_INDEX      = "messages"
CONVERSATIONS_INDEX = "conversations"
USERS_INDEX         = "users"
ES_HOST             = os.getenv("ELASTICSEARCH_HOST")
ES_API_KEY          = os.getenv("ELASTICSEARCH_API_KEY")

# ──────────────────────────────────────────────
# Index Mappings
# ──────────────────────────────────────────────

USERS_MAPPING = {
    "mappings": {
        "properties": {
            "name":            {"type": "keyword"},
            "registered_at":   {"type": "date"}
        }
    }
}

MESSAGES_MAPPING = {
    "mappings": {
        "properties": {
            "conversation_id": {"type": "keyword"},
            "from_user":       {"type": "keyword"},
            "to_user":         {"type": "keyword"},
            "text":            {"type": "text"},
            "timestamp":       {"type": "date"}
        }
    }
}

CONVERSATIONS_MAPPING = {
    "mappings": {
        "properties": {
            "conversation_id": {"type": "keyword"},
            "from_user":       {"type": "keyword"},
            "to_user":         {"type": "keyword"},
            "text":            {"type": "text"},
            "timestamp":       {"type": "date"},
            "embedding": {
                "type":       "dense_vector",
                "dims":       1536,
                "index":      True,
                "similarity": "cosine"
            }
        }
    }
}

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

def create_index_if_not_exists(es: Elasticsearch, index_name: str, mapping: dict):
    if es.indices.exists(index=index_name):
        print(f"  ⏭️  Index '{index_name}' already exists — skipping.")
    else:
        es.indices.create(index=index_name, body=mapping)
        print(f"  ✅  Index '{index_name}' created successfully.")


def main():
    print("\n🔌 Connecting to Elasticsearch (Elastic Cloud)...")
    es = Elasticsearch(ES_HOST, api_key=ES_API_KEY)

    if not es.ping():
        print("  ❌  Could not connect to Elasticsearch at", ES_HOST)
        print("      Make sure Elasticsearch is running and try again.")
        return

    print(f"  ✅  Connected to Elasticsearch at {ES_HOST}\n")
    print("📦 Setting up indices...\n")

    create_index_if_not_exists(es, USERS_INDEX,         USERS_MAPPING)
    create_index_if_not_exists(es, MESSAGES_INDEX,      MESSAGES_MAPPING)
    create_index_if_not_exists(es, CONVERSATIONS_INDEX, CONVERSATIONS_MAPPING)

    print("\n🎉 Setup complete! You can now start the app:\n")
    print("    uvicorn main:app --reload\n")


if __name__ == "__main__":
    main()
