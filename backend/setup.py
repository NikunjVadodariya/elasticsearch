import os
from pathlib import Path
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ──────────────────────────────────────────────
# Constants — index names used across the app
# ──────────────────────────────────────────────
MESSAGES_INDEX      = "messages"
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

EMBEDDING_MAPPING = {
    "type":       "dense_vector",
    "dims":       1536,
    "index":      True,
    "similarity": "cosine",
}

# Autocomplete field: ES analyzes the text AT INDEX TIME into edge n-grams
# (p, pa, pas, pass, ...) and word shingles; the query is analyzed with the same analyzer.
SAYT_MAPPING = {"type": "search_as_you_type"}

MESSAGES_MAPPING = {
    "mappings": {
        "properties": {
            "conversation_id": {"type": "keyword"},
            "from_user":       {"type": "keyword"},
            "to_user":         {"type": "keyword"},
            "text":            {"type": "text"},
            "timestamp":       {"type": "date"},
            "embedding":       EMBEDDING_MAPPING,
            "text_sayt":       SAYT_MAPPING
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


def ensure_message_fields(es: Elasticsearch):
    """Add new fields to an existing `messages` index and backfill autocomplete data."""
    es.indices.put_mapping(
        index=MESSAGES_INDEX,
        properties={"embedding": EMBEDDING_MAPPING, "text_sayt": SAYT_MAPPING},
    )
    es.update_by_query(
        index=MESSAGES_INDEX,
        query={"bool": {"must_not": {"exists": {"field": "text_sayt"}}}},
        script={"source": "ctx._source.text_sayt = ctx._source.text", "lang": "painless"},
        conflicts="proceed",
        refresh=True,
    )


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
    # Older `messages` indexes have no embedding field yet — add it (no delete needed)
    ensure_message_fields(es)

    print("\n🎉 Setup complete! You can now start the app:\n")
    print("    uvicorn main:app --reload\n")


if __name__ == "__main__":
    main()
