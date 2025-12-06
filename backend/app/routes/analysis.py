from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import io
from PIL import Image
import logging

# 로거 설정
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analysis"])

class AnalysisResponse(BaseModel):
    detection: Dict
    symptoms: Dict
    region_details: list

class AnalyzeAndRecommendResponse(BaseModel):
    analysis: AnalysisResponse
    recommendations: Dict

def validate_image(file: UploadFile) -> bool:
    """
    이미지 파일 유효성 검증
    
    검증 항목:
    - 파일 크기 (10MB 이하)
    - 파일 형식 (JPG, JPEG, PNG)
    - 실제 이미지 파일인지 확인
    
    Raises:
        HTTPException: 검증 실패 시
    """
    try:
        # 1. 파일 크기 체크 (10MB)
        file.file.seek(0, 2)  # 파일 끝으로 이동
        file_size = file.file.tell()
        file.file.seek(0)  # 처음으로 되돌림
        
        if file_size == 0:
            logger.warning("빈 파일 업로드 시도")
            raise HTTPException(status_code=400, detail="빈 파일입니다. 이미지를 선택해주세요.")
        
        if file_size > 10 * 1024 * 1024:
            logger.warning(f"파일 크기 초과: {file_size} bytes")
            raise HTTPException(status_code=400, detail="파일 크기는 10MB 이하여야 합니다.")
        
        # 2. 파일 형식 체크
        if not file.filename:
            raise HTTPException(status_code=400, detail="파일 이름이 없습니다.")
        
        allowed_formats = ['jpg', 'jpeg', 'png']
        ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        
        if ext not in allowed_formats:
            logger.warning(f"지원하지 않는 파일 형식: {ext}")
            raise HTTPException(
                status_code=400, 
                detail=f"JPG, JPEG, PNG 파일만 업로드 가능합니다. (현재: {ext})"
            )
        
        # 3. 실제 이미지 파일인지 확인
        try:
            image_bytes = file.file.read()
            file.file.seek(0)
            
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()
            
            # 이미지 크기 체크
            width, height = image.size
            if width < 100 or height < 100:
                raise HTTPException(
                    status_code=400,
                    detail="이미지 해상도가 너무 낮습니다. (최소 100x100 픽셀)"
                )
            
            logger.info(f"파일 검증 성공: {file.filename} ({file_size} bytes, {width}x{height})")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"이미지 검증 실패: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail="유효하지 않은 이미지 파일입니다. 다른 이미지를 시도해주세요."
            )
        
        return True
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 검증 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="파일 검증 중 오류가 발생했습니다.")

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_skin(file: UploadFile = File(...)):
    """
    얼굴 이미지 피부 분석
    
    - Detection: 얼굴 7개 부위 검출
    - Classification: 4개 증상 분류
    """
    from ..services.analyzer import SkinAnalyzer
    
    try:
        logger.info(f"분석 요청: {file.filename}")
        
        # 파일 검증
        validate_image(file)
        
        # 이미지 읽기
        image_bytes = await file.read()
        
        # 분석 실행
        analyzer = SkinAnalyzer()
        result = analyzer.analyze(image_bytes)
        
        # 분석 실패 체크
        if not result.get('success', True):
            error_msg = result.get('error_message', '분석에 실패했습니다.')
            logger.warning(f"분석 실패: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        logger.info("분석 성공")
        return AnalysisResponse(
            detection=result['detection'],
            symptoms=result['symptoms'],
            region_details=result['region_details']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"분석 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        )

@router.post("/analyze-and-recommend")
async def analyze_and_recommend(
    file: UploadFile = File(...),
    skin_type: str = Form(...),
    budget_min: int = Form(0),
    budget_max: int = Form(1000000),
    category: Optional[str] = Form(None),
    query: Optional[str] = Form(None)
):
    """
    이미지 분석 + 제품 추천 통합
    
    1. 피부 분석
    2. 분석 결과 기반 제품 추천
    """
    from ..services.analyzer import SkinAnalyzer
    from ..services.recommender import RecommendationEngine
    
    try:
        logger.info(f"분석+추천 요청: {file.filename}, 피부타입={skin_type}, 예산={budget_min}-{budget_max}")
        
        # 파일 검증
        validate_image(file)
        
        # 입력 값 검증
        valid_skin_types = ['복합성', '지성', '건성', '민감성', '여드름성']
        if skin_type not in valid_skin_types:
            raise HTTPException(
                status_code=400,
                detail=f"유효하지 않은 피부 타입입니다. ({', '.join(valid_skin_types)} 중 선택)"
            )
        
        if budget_min < 0 or budget_max < 0:
            raise HTTPException(status_code=400, detail="예산은 0원 이상이어야 합니다.")
        
        if budget_min > budget_max:
            raise HTTPException(status_code=400, detail="최소 예산이 최대 예산보다 클 수 없습니다.")
        
        # Step 1: 피부 분석
        try:
            image_bytes = await file.read()
            analyzer = SkinAnalyzer()
            analysis_result = analyzer.analyze(image_bytes)
            
            # 분석 실패 체크
            if not analysis_result.get('success', True):
                error_msg = analysis_result.get('error_message', '분석에 실패했습니다.')
                logger.warning(f"피부 분석 실패: {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)
            
            logger.info(f"피부 분석 완료: {analysis_result['symptoms']}")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"피부 분석 중 오류: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="피부 분석 중 오류가 발생했습니다. 다시 시도해주세요."
            )
        
        # Step 2: 제품 추천
        try:
            engine = RecommendationEngine()
            recommendations = await engine.recommend(
                skin_analysis=analysis_result['symptoms'],
                user_preferences={
                    'skin_type': skin_type,
                    'budget_min': budget_min,
                    'budget_max': budget_max,
                    'category': category
                },
                query=query
            )
            
            logger.info(f"제품 추천 완료: {len(recommendations.get('products', []))}개")
            
        except Exception as e:
            logger.error(f"제품 추천 중 오류: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="제품 추천 중 오류가 발생했습니다. 다시 시도해주세요."
            )
        
        # 성공 응답
        return {
            'analysis': {
                'detection': analysis_result['detection'],
                'symptoms': analysis_result['symptoms'],
                'region_details': analysis_result['region_details']
            },
            'recommendations': recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"전체 프로세스 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        )