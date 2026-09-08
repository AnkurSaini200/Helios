import asyncio
import logging
import os
from contextlib import asynccontextmanager
# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles

from app.database.session import Base, SessionLocal, engine
from app.routes.analytics import router as analytics_router
from app.routes.buses import router as buses_router
from app.routes.detect import router as detect_router
from app.routes.video_pipeline import router as video_pipeline_router
from app.routes.dashcam_pipeline import router as dashcam_router
from app.routes.gps import router as gps_router
from app.routes.health import router as health_router
from app.routes.incidents import router as incidents_router
from app.routes.mock import router as mock_router
from app.routes.models import router as models_router
from app.routes.notifications import router as notifications_router
from app.routes.sos import router as sos_router
from app.routes.ws import router as ws_router
from app.services.seed_data import seed_database
from app.services.simulator import run_simulation_loop

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("helios.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing HELIOS Database Tables...")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        seed_database(db)
        logger.info("HELIOS Database initialized and seeded successfully.")
    except Exception as e:
        logger.error(f"Error seeding database: {e}")
    finally:
        db.close()

    # Start mock simulator task
    simulator_task = asyncio.create_task(run_simulation_loop())

    yield

    # Shutdown
    simulator_task.cancel()
    try:
        await simulator_task
    except asyncio.CancelledError:
        pass
    logger.info("HELIOS Server shut down cleanly.")


app = FastAPI(
    title="HELIOS Central Command API",
    description="AI-Powered Smart City Road Intelligence & Emergency Response Platform. "
                "Aggregates mobile dashcam edge AI detections from electric city buses.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount media static directory for real edge detection images
media_dir = os.path.join(os.path.dirname(__file__), "media")
os.makedirs(media_dir, exist_ok=True)
app.mount("/media", StaticFiles(directory=media_dir), name="media")

# API v1 Router Registration
API_PREFIX = "/api/v1"

app.include_router(health_router, prefix=API_PREFIX)
app.include_router(health_router, prefix="/api")  # backward-compat /api/health
app.include_router(buses_router, prefix=API_PREFIX)
app.include_router(gps_router, prefix=API_PREFIX)
app.include_router(incidents_router, prefix=API_PREFIX)
app.include_router(detect_router, prefix=API_PREFIX)
app.include_router(video_pipeline_router, prefix=API_PREFIX)
app.include_router(dashcam_router, prefix=API_PREFIX)
app.include_router(sos_router, prefix=API_PREFIX)
app.include_router(models_router, prefix=API_PREFIX)
app.include_router(analytics_router, prefix=API_PREFIX)
app.include_router(notifications_router, prefix=API_PREFIX)
app.include_router(mock_router, prefix=API_PREFIX)
app.include_router(ws_router, prefix=API_PREFIX)
app.include_router(ws_router)  # expose /ws/events at root


@app.get("/")
def read_root():
    return {
        "name": "HELIOS Central Command Platform",
        "tagline": "Intelligent Mobility. Safer Cities.",
        "status": "operational",
        "version": "1.0.0",
        "docs": "/docs",
        "api_v1": "/api/v1",
        "websocket": "/api/v1/ws/events",
    }