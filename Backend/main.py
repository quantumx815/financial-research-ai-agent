from routes.company import router as company_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Financial Research AI Agent",
    description="AI-powered financial research backend",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(company_router, prefix="/api")


@app.get("/")
def root():
    return {
        "message": "Financial Research AI Agent API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


@app.get("/api/test")
def test_connection():
    return {
        "message": "Hello from Financial Research AI Backend"
    }