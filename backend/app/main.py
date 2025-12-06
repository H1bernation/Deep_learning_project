from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Skin Analysis API")

# CORS 설정 (프론트엔드 연결용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
from app.routes.recommendation import router as recommendation_router
from app.routes.analysis import router as analysis_router
from app.routes.chat import router as chat_router

app.include_router(recommendation_router)
app.include_router(analysis_router)
app.include_router(chat_router) 

@app.get("/")
def read_root():
    return {
        "message": "Skin Analysis API is running!",
        "status": "healthy"
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}