from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

router = APIRouter(prefix="/api/v1", tags=["recommendation"])

class SkinAnalysis(BaseModel):
    wrinkle: int = Field(..., ge=0, le=3, description="주름 심각도")
    pore: int = Field(..., ge=0, le=3, description="모공 심각도")
    pigmentation: int = Field(..., ge=0, le=3, description="색소 침착")
    sagging: int = Field(..., ge=0, le=3, description="피부 처짐")

class UserPreferences(BaseModel):
    skin_type: Literal["지성", "건성", "복합성", "민감성"]
    budget_min: int = Field(default=0, ge=0)
    budget_max: int = Field(default=1000000, ge=0)
    category: Optional[Literal["토너", "로션", "에센스/세럼/앰플"]] = None

class RecommendationRequest(BaseModel):
    skin_analysis: SkinAnalysis
    user_preferences: UserPreferences
    query: Optional[str] = Field(None, max_length=200)

class ProductRecommendation(BaseModel):
    id: int
    name: str
    brand: str
    price: int
    category: str
    score: float
    explanation: str
    effects: dict

class RecommendationResponse(BaseModel):
    products: List[ProductRecommendation]
    total_candidates: int
    search_time_ms: int

@router.post("/recommend", response_model=RecommendationResponse)
async def recommend_products(request: RecommendationRequest):
    from ..services.recommender import RecommendationEngine
    import time
    
    start_time = time.time()
    
    try:
        engine = RecommendationEngine()
        results = await engine.recommend(
            skin_analysis=request.skin_analysis.dict(),
            user_preferences=request.user_preferences.dict(),
            query=request.query
        )
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return RecommendationResponse(
            products=results['products'],
            total_candidates=results['total_candidates'],
            search_time_ms=elapsed_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))