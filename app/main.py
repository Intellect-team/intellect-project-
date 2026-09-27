from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager

from app.api.routes import router
from app.services import classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the YOLO classifier once at startup so the first /analyze isn't slow
    # and a broken model file shows up in the logs immediately.
    status = classifier.model_status()
    print(f"[classifier] loaded={status['loaded']} classes={status['classes']} error={status['error']}")
    yield


app = FastAPI(title="Isaafi Backend", lifespan=lifespan)

# Restrict this to your actual frontend origin(s) before going beyond a demo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(router)
