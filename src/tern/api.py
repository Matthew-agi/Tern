"""Public Tern identity around the byte-preserved scoring API."""
import os
from fastapi import HTTPException
import serve_clef_api as frozen

frozen.NAME = os.environ.get("TERN_SERVED_MODEL_NAME", "tern")
app = frozen.app
app.title = "Tern"


@app.get("/ready")
async def ready():
    # Launcher creates this only after a real synthetic inference succeeds.
    from pathlib import Path
    if not Path(os.environ["TERN_READY_FILE"]).exists():
        raise HTTPException(503, "Tern is warming up")
    return await frozen.health()
