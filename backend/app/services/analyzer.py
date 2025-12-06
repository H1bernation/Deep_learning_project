from .detector import FaceDetectionService
from .classifier import SkinClassificationService
from typing import Dict
import logging

# 로거 설정
logger = logging.getLogger(__name__)

class SkinAnalyzer:
    def __init__(self):
        """
        피부 분석기 초기화
        
        Raises:
            RuntimeError: 모델 로드 실패 시
        """
        try:
            self.detector = FaceDetectionService()
            logger.info("Detection 모델 로드 완료")
        except Exception as e:
            logger.error(f"Detection 모델 로드 실패: {str(e)}")
            raise RuntimeError(f"얼굴 검출 모델을 로드할 수 없습니다: {str(e)}")
        
        try:
            self.classifier = SkinClassificationService()
            logger.info("Classification 모델 로드 완료")
        except Exception as e:
            logger.error(f"Classification 모델 로드 실패: {str(e)}")
            raise RuntimeError(f"피부 분석 모델을 로드할 수 없습니다: {str(e)}")
    
    def analyze(self, image_bytes: bytes) -> Dict:
        """
        전체 피부 분석 파이프라인
        
        1. Detection: 얼굴 부위 검출
        2. Classification: 각 부위별 증상 분류
        3. Aggregation: 전체 증상 점수 계산
        
        Args:
            image_bytes: 이미지 바이트 데이터
        
        Returns:
            {
                'success': bool,
                'error_message': str (에러 시),
                'detection': {'boxes': [...], 'image_shape': (h, w)},
                'symptoms': {'wrinkle': 2, 'pore': 1, ...},
                'region_details': [...]
            }
        """
        default_symptoms = {'wrinkle': 0, 'pore': 0, 'pigmentation': 0, 'sagging': 0}
        
        try:
            # Step 0: 이미지 데이터 유효성 검사
            if not image_bytes or len(image_bytes) == 0:
                logger.warning("빈 이미지 데이터")
                return {
                    'success': False,
                    'error_message': '이미지 데이터가 비어있습니다.',
                    'symptoms': default_symptoms,
                    'detection': {'boxes': [], 'image_shape': (0, 0)},
                    'region_details': []
                }
            
            if len(image_bytes) > 10 * 1024 * 1024:  # 10MB
                logger.warning(f"이미지 크기 초과: {len(image_bytes)} bytes")
                return {
                    'success': False,
                    'error_message': '이미지 크기는 10MB 이하여야 합니다.',
                    'symptoms': default_symptoms,
                    'detection': {'boxes': [], 'image_shape': (0, 0)},
                    'region_details': []
                }
            
            # Step 1: Detection
            try:
                detection_result = self.detector.detect(image_bytes)
                logger.info(f"Detection 완료: {len(detection_result['boxes'])}개 부위 검출")
            except Exception as e:
                logger.error(f"Detection 실패: {str(e)}")
                return {
                    'success': False,
                    'error_message': f'얼굴 검출 중 오류가 발생했습니다. 다른 이미지를 시도해주세요.',
                    'symptoms': default_symptoms,
                    'detection': {'boxes': [], 'image_shape': (0, 0)},
                    'region_details': []
                }
            
            # 얼굴 검출 실패 처리
            if not detection_result['boxes'] or len(detection_result['boxes']) == 0:
                logger.warning("얼굴 부위 검출 실패")
                return {
                    'success': False,
                    'error_message': '얼굴 부위를 검출하지 못했습니다. 정면 얼굴 사진을 업로드해주세요.',
                    'detection': detection_result,
                    'symptoms': default_symptoms,
                    'region_details': []
                }
            
            # Step 2: Classification
            try:
                symptoms = self.classifier.classify_all_regions(
                    image_bytes, 
                    detection_result['boxes']
                )
                logger.info(f"Classification 완료: {symptoms}")
            except Exception as e:
                logger.error(f"Classification 실패: {str(e)}")
                return {
                    'success': False,
                    'error_message': f'피부 증상 분석 중 오류가 발생했습니다. 다시 시도해주세요.',
                    'detection': detection_result,
                    'symptoms': default_symptoms,
                    'region_details': []
                }
            
            # Step 3: 부위별 상세 정보
            region_details = []
            failed_regions = 0
            
            for box_info in detection_result['boxes']:
                try:
                    region_symptom = self.classifier.classify_region(
                        image_bytes,
                        box_info['bbox']
                    )
                    region_details.append({
                        'region': box_info['class'],
                        'bbox': box_info['bbox'],
                        'confidence': box_info['confidence'],
                        'symptoms': region_symptom
                    })
                except Exception as e:
                    # 개별 부위 분석 실패는 경고만 하고 계속 진행
                    logger.warning(f"부위 '{box_info['class']}' 분석 실패: {str(e)}")
                    failed_regions += 1
                    continue
            
            if failed_regions > 0:
                logger.info(f"{failed_regions}개 부위 분석 실패, {len(region_details)}개 부위 분석 성공")
            
            # 최소 1개 이상의 부위가 분석되었는지 확인
            if len(region_details) == 0:
                logger.error("모든 부위 분석 실패")
                return {
                    'success': False,
                    'error_message': '피부 부위 분석에 실패했습니다. 다른 이미지를 시도해주세요.',
                    'detection': detection_result,
                    'symptoms': default_symptoms,
                    'region_details': []
                }
            
            # 성공 응답
            logger.info("피부 분석 완료")
            return {
                'success': True,
                'detection': detection_result,
                'symptoms': symptoms,
                'region_details': region_details
            }
            
        except Exception as e:
            # 예상치 못한 오류
            logger.error(f"예상치 못한 오류: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error_message': f'분석 중 예상치 못한 오류가 발생했습니다. 관리자에게 문의하세요.',
                'symptoms': default_symptoms,
                'detection': {'boxes': [], 'image_shape': (0, 0)},
                'region_details': []
            }