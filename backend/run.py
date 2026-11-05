import uvicorn
from setup import main as run_setup

if __name__ == "__main__":
    # Step 1 — Create ES indices (skips if already exist)
    run_setup()

    # Step 2 — Start FastAPI app
    print("🌐 Starting server at http://localhost:8000\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
