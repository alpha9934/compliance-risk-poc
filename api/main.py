from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import score, cases, audit

app = FastAPI(
    title="Compliance Risk Prediction API",
    description="AI-powered financial compliance risk scoring — Deloitte POC by The Talent Grid",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(score.router)
app.include_router(cases.router)
app.include_router(audit.router)


@app.get("/", tags=["health"])
async def root():
    return {
        "service": "Compliance Risk Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}
