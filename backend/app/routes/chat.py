from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional

router = APIRouter(prefix="/api/v1", tags=["chat"])

class ChatRequest(BaseModel):
    question: str
    analysis: Dict
    recommendations: List[Dict]
    user_preferences: Optional[Dict] = None

class ChatResponse(BaseModel):
    answer: str
    context_used: Dict

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    RAG 기반 챗봇
    
    - 피부 분석 결과와 추천 제품을 기반으로 질문에 답변
    - 유사 리뷰 검색 및 제품 상세 정보 활용
    """
    from ..services.rag_service import RAGService
    
    try:
        rag_service = RAGService()
        
        answer = rag_service.generate_response(
            question=request.question,
            analysis=request.analysis,
            recommendations=request.recommendations,
            user_preferences=request.user_preferences
        )
        
        return ChatResponse(
            answer=answer,
            context_used={
                "products_count": len(request.recommendations),
                "analysis": request.analysis
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))