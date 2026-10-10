
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.config import FRONTEND_ORIGINS

app=FastAPI(title="Netflix Insight Studio API",version="1.0.0",
            description="Analytics, recommendations and educational machine learning API for the supplied Netflix metadata dataset.")
app.add_middleware(CORSMiddleware,allow_origins=FRONTEND_ORIGINS,allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
app.include_router(router)

@app.get("/")
def root():
    return {"name":"Netflix Insight Studio","docs":"/docs","api":"/api/health"}
