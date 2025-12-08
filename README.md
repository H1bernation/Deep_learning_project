# CLORO - AI 피부 분석 및 스킨케어 추천 시스템

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)

**Classification-based Learning for Observation & Recommendation Optimization**

딥러닝 기반 얼굴 피부 분석 및 맞춤형 화장품 추천 서비스

</div>

---

## 📋 목차

1. [프로젝트 개요](#-프로젝트-개요)
2. [시스템 아키텍처](#-시스템-아키텍처)
3. [주요 기능](#-주요-기능)
4. [기술 스택](#-기술-스택)
5. [성능 지표](#-성능-지표)
6. [설치 및 실행](#-설치-및-실행)
7. [프로젝트 구조](#-프로젝트-구조)
8. [API 문서](#-api-문서)
9. [개발 과정](#-개발-과정)
10. [참고 문헌](#-참고-문헌)

---

## 🎯 프로젝트 개요

**CLORO**는 컴퓨터 비전과 자연어 처리 기술을 결합하여 사용자의 피부 상태를 분석하고, 맞춤형 화장품을 추천하는 AI 기반 스킨케어 서비스입니다.

### 핵심 가치
- **정확한 분석**: YOLOv8 + Ensemble CNN으로 얼굴 7개 부위, 4개 증상 분석
- **개인화 추천**: 피부 타입, 예산, 증상을 고려한 3단계 하이브리드 추천
- **대화형 상담**: RAG 기반 챗봇으로 실시간 피부 관리 상담

---

## 🏗️ 시스템 아키텍처
```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                             │
│                    Streamlit Web App                         │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/REST API
┌──────────────────────▼──────────────────────────────────────┐
│                      Backend (FastAPI)                       │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Detection  │→ │Classification│→ │  Recommendation  │   │
│  │  (YOLOv8)   │  │  (Ensemble)  │  │  (Hybrid System) │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         RAG Chatbot (GPT-4o-mini + Qdrant)          │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────┬───────────────────┬──────────────────────────┘
                │                   │
        ┌───────▼──────┐    ┌──────▼──────┐
        │  PostgreSQL  │    │   Qdrant    │
        │  (Products)  │    │  (Vectors)  │
        └──────────────┘    └─────────────┘
```

### 분석 파이프라인
```
Input Image
    ↓
[1] Face Detection (YOLOv8)
    → 7개 부위 검출: 이마, 미간, 눈가(L/R), 볼(L/R), 턱
    ↓
[2] Skin Classification (Ensemble)
    → Vision Transformer + EfficientNetV2
    → 4개 증상 분류: 주름, 모공, 색소침착, 탄력저하
    ↓
[3] Symptom Aggregation
    → 부위별 최대값 기반 집계
    ↓
[4] Product Recommendation (3단계 하이브리드)
    → Stage 1: Hard Filtering (피부타입, 예산, 카테고리)
    → Stage 2: Symptom-Weighted Matching (증상-효능 매칭)
    → Stage 3: Semantic Similarity (Qdrant 벡터 검색)
    → 카테고리 균형 출력 (토너, 로션, 세럼 각 3개)
    ↓
Output: 분석 결과 + 추천 제품
```

---

## ✨ 주요 기능

### 1. 피부 분석 (Skin Analysis)
- **얼굴 부위 검출**: YOLOv8 기반 7개 영역 자동 감지
- **증상 분류**: ViT + EfficientNetV2 앙상블로 4개 증상 정밀 분석
- **시각화**: 레이더 차트로 증상 심각도 직관적 표시

### 2. 제품 추천 (Product Recommendation)
- **3단계 하이브리드 시스템**:
  1. **Hard Filtering**: 피부 타입, 예산, 카테고리 균형 필터링
  2. **Symptom-Weighted Matching**: 증상 심각도 × 제품 효과 가중 매칭
  3. **Semantic Similarity**: Qdrant 벡터 DB로 리뷰 의미 유사도 검색
- **최종 점수**: `F = 0.4×증상매칭 + 0.3×의미유사도 + 0.2×리뷰평점 + 0.1×가격효율`
- **카테고리 균형**: 토너/로션/세럼 각 3개씩 추천

### 3. AI 상담 챗봇 (RAG Chatbot)
- **컨텍스트 기반 답변**: 분석 결과 + 제품 정보 + 유사 리뷰 활용
- **실시간 대화**: GPT-4o-mini 기반 자연스러운 상담
- **벡터 검색**: Qdrant로 관련 리뷰 빠른 검색

---

## 🛠️ 기술 스택

### Backend
| 카테고리 | 기술 | 버전 | 용도 |
|---------|------|------|------|
| Framework | FastAPI | 0.104.1 | REST API 서버 |
| Deep Learning | PyTorch | 2.0+ | 모델 추론 |
| Detection | Ultralytics YOLOv8 | 8.3+ | 얼굴 부위 검출 |
| Classification | timm (ViT, EfficientNetV2) | 1.0+ | 피부 증상 분류 |
| Vision | OpenCV | 4.0+ | 이미지 전처리 |
| Database | PostgreSQL | 16 | 제품 정보 저장 |
| Vector DB | Qdrant | 1.7+ | 임베딩 벡터 저장 |
| AI API | OpenAI | - | GPT-4o-mini 챗봇 |

### Frontend
| 카테고리 | 기술 | 버전 | 용도 |
|---------|------|------|------|
| Framework | Streamlit | 1.51.0 | 웹 UI |
| Visualization | Plotly | 6.0+ | 차트 및 그래프 |
| HTTP | Requests | 2.31+ | API 호출 |

### Infrastructure
- **Docker Compose**: 멀티 컨테이너 관리

---

## 📊 성능 지표

### Detection Model (YOLOv8)
```
Dataset: AI Hub 한국인 안면 피부 상태 측정 데이터 (13,936장)
Classes: 7개 (이마, 미간, 눈가L, 눈가R, 볼L, 볼R, 턱)

Results:
├── mAP50: 0.992
├── mAP50-95: 0.761
├── Precision: 0.979
└── Recall: 0.981
```

### Classification Model (Ensemble)
```
Dataset: AI Hub 안면 피부 상태 측정 데이터
Symptoms: 4개 (주름, 모공, 색소침착, 탄력저하)
Severity: 4단계 (0: 없음, 1: 경미, 2: 중등도, 3: 심각)

Model Architecture:
├── Vision Transformer (vit_base_patch16_224)
├── EfficientNetV2-M (efficientnetv2_m)
└── Ensemble: Soft Voting

Results:
├── Accuracy: 0.8592
├── F1-Score: 0.8572
└── 목표(0.85) 대비: +0.85% 초과 달성
```

### Recommendation System
```
Database: Olive Young 제품 360개
Algorithm: 3-Stage Hybrid (Hard Filtering → Symptom Matching → Semantic Search)

Metrics:
├── 평균 응답 시간: < 2초
├── 카테고리 균형도: 100% (토너/로션/세럼 각 3개)
└── 예산 준수율: 100%
```

---

## 🚀 설치 및 실행

### 사전 요구사항

- **Docker Desktop** 설치 필수
- **Git** (코드 관리)
- **OpenAI API Key** (챗봇 기능용)

### 1. 레포지토리 클론
```bash
git clone https://github.com/[사용자명]/CLORO.git
cd CLORO
```

### 2. 모델 파일 다운로드

**Google Drive 링크**: https://drive.google.com/drive/folders/1MaVcJALQvaXCPnRKGwvwePX7jlW1aT4K?usp=drive_link

다운로드 후 아래 경로에 배치:
```
backend/models/
├── detection.pt           # YOLOv8 얼굴 부위 검출 모델
└── classification.pth     # ViT + EfficientNetV2 앙상블 분류 모델
```

### 3. 환경 변수 설정
```bash
# .env.example을 복사하여 .env 생성
cp .env.example .env

# .env 파일 편집하여 OpenAI API Key 입력
```

**.env 파일 내용:**
```env
OPENAI_API_KEY=sk-proj-여기에_실제_키_입력
```

### 4. Docker 컨테이너 실행
```bash
# 빌드 및 실행
docker compose up -d --build

# 로그 확인
docker compose logs -f
```

### 5. 서비스 접속

| 서비스 | URL | 설명 |
|--------|-----|------|
| Frontend | http://localhost:8501 | 사용자 인터페이스 |
| Backend API | http://localhost:8000/docs | Swagger UI 문서 |
| PostgreSQL | localhost:5433 | 데이터베이스 |
| Qdrant | localhost:6334 | 벡터 DB 대시보드 |

### 6. 서비스 종료
```bash
# 중지
docker compose stop

# 완전 삭제 (데이터 포함)
docker compose down -v
```

---

## 📁 프로젝트 구조
```
CLORO/
├── backend/                      # FastAPI 백엔드
│   ├── app/
│   │   ├── routes/              # API 엔드포인트
│   │   │   ├── analysis.py      # 분석 API
│   │   │   └── chat.py          # 챗봇 API
│   │   ├── services/            # 비즈니스 로직
│   │   │   ├── detector.py      # YOLOv8 Detection
│   │   │   ├── classifier.py    # Ensemble Classification
│   │   │   ├── analyzer.py      # 분석 통합
│   │   │   ├── recommender.py   # 추천 시스템
│   │   │   └── rag_service.py   # RAG 챗봇
│   │   └── main.py              # FastAPI 앱
│   ├── models/                   # 모델 파일
│   │   ├── detection.pt
│   │   └── classification.pth
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                     # Streamlit 프론트엔드
│   ├── .streamlit/
│   │   └── config.toml
│   ├── contents/                 # 미디어 파일
│   ├── fonts/                    # 커스텀 폰트
│   ├── app.py                    # 메인 UI
│   ├── styles.css
│   ├── requirements.txt
│   └── Dockerfile
│
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## 📚 API 문서

상세한 API 문서는 [Swagger UI](http://localhost:8000/docs)에서 확인하세요.

**주요 엔드포인트:**
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/analyze` | 피부 분석 |
| POST | `/api/v1/analyze-and-recommend` | 분석 + 추천 |
| POST | `/api/v1/chat` | AI 챗봇 |

---

## 🔬 개발 과정

| 주차 | 내용 |
|------|------|
| Week 1-2 | 데이터 수집 (AI Hub 13,936장, Olive Young 360개) 및 전처리 |
| Week 3-4 | YOLOv8 Detection 모델 개발 (mAP50: 0.992) |
| Week 5-6 | ViT + EfficientNetV2 앙상블 분류 모델 (F1: 0.8572) |
| Week 7 | 3단계 하이브리드 추천 시스템 + RAG 챗봇 |
| Week 8 | FastAPI/Streamlit 통합, Docker 컨테이너화 |

---

## 📜 참고 문헌

- [1] Lee, J., et al. (2024). "Deep learning-based skin care product recommendation system using facial image analysis"
- [2] Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- [3] Tan, M., & Le, Q. (2021). "EfficientNetV2: Smaller Models and Faster Training"
- [4] Ganaie, M. A., et al. (2022). "Ensemble deep learning: A review"
- [5] Redmon, J., et al. (2016). "You Only Look Once: Unified, Real-Time Object Detection"

---

## 👤 개발자

**동민**
- GitHub: https://github.com/H1bernation
- Email: hdm1023@sju.ac.kr
- 프로젝트 기간: 2025.10 - 2025.12

---

## 🙏 감사의 말

- **AI Hub**: 한국인 안면 피부 상태 측정 데이터 제공
- **Olive Young**: 제품 정보 참조

---

## 📝 라이센스

MIT License

---

<div align="center">

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요! ⭐**

</div>