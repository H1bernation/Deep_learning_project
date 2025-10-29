"#My project" 


-----

# **피부 진단 및 화장품 추천 AI 시스템 (가칭: CROLO)**

## 🌟 1. 프로젝트 소개 및 목표

최신 딥러닝 기술을 활용하여 사용자 맞춤형 피부 진단 및 화장품 추천 서비스를 제공하는 AI 기반 통합 시스템입니다.

  * **핵심 목표**: 얼굴 사진 분석 ($\text{YOLOv8}$ 및 $\text{ViT}$ 앙상블)을 통해 **객관적인 피부 상태**를 진단하고, $\text{RAG}$ 시스템을 통해 **실사용자 리뷰 기반의 신뢰성 있는 제품**을 추천합니다.
  * **차별점**: 주관적인 설문 방식의 한계를 극복하고, $\text{AI}$의 진단 결과와 $\text{LLM}$의 대화 능력을 결합하여 **정밀하고 유연한 맞춤형 상담**을 제공합니다.

## 🛠️ 2. 기술 스택 (Tech Stack)

프로젝트를 구현하는 데 사용된 주요 기술 목록입니다.

| 구분 | 기술 스택 | 역할 |
| :--- | :--- | :--- |
| **AI 모델** | YOLOv8, ViT, EfficientNetV2 | 얼굴 부위 탐지 및 4대 피부 증상 분류 (주름, 모공, 탄력, 색소침착) |
| **백엔드/API** | FastAPI, Docker Compose | AI 모델 서빙, 비동기 API 엔드포인트 제공 및 컨테이너 오케스트레이션 |
| **프론트엔드** | Streamlit | Python 기반의 웹 사용자 인터페이스 및 챗봇 환경 구현 |
| **데이터/LLM** | Qdrant (Vector DB), GPT-4o-mini, LangChain | 리뷰 벡터 저장 및 고속 검색, $\text{RAG}$ 시스템 구축 및 대화 처리 |

## ⚙️ 3. 설치 및 실행 방법 (Quick Start)

본 프로젝트를 로컬 환경에서 실행하기 위한 간단한 절차입니다. **Docker Compose**를 사용해 모든 컨테이너를 한 번에 실행합니다.

### 1\) 필수 환경 설정

  * **Docker 및 Docker Compose 설치**
  * **Python 3.10 이상** 환경 준비

### 2\) 환경 변수 설정

루트 경로에 `.env` 파일을 생성하고 다음 항목을 입력합니다. (API Key는 $\text{LLM}$ 사용에 필요합니다.)

```bash
# OpenAI API Key (GPT-4o-mini 사용 시)
OPENAI_API_KEY="YOUR_API_KEY"
```

### 3\) Docker Compose 실행

프로젝트 루트 폴더에서 다음 명령어를 실행하여 모든 컨테이너를 빌드하고 실행합니다.

```bash
docker compose up --build
```

### 4\) 접속 확인

  * **웹 서비스 (Streamlit)**: `localhost:8501`
  * **API 서버 (FastAPI Docs)**: `localhost:8000/docs`

## 📊 4. 프로젝트 구조 (Directory Structure)

```
.
├── app/                  # FastAPI 및 Streamlit 애플리케이션 코드
│   ├── api/              # FastAPI 엔드포인트 정의 (detect, classify, recommend 등)
│   ├── streamlit_app.py  # Streamlit UI 코드 및 챗봇 인터페이스
├── data/                 # 데이터 전처리 스크립트 및 라벨 파일 (.txt)
├── models/               # 학습된 AI 모델 가중치 파일 (.pt, .pth)
├── requirements/         # 각 컨테이너별 Python 종속성 파일
└── docker-compose.yml    # Docker 설정 파일
```

## 📜 5. 참고 문헌

프로젝트의 학술적 근거가 된 주요 논문입니다.

  * [1] Lee, J., et al. (2024). "Deep learning-based skin care product recommendation..."
  * [2] Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words..."
  * [3] Tan, M., & Le, Q. (2021). "EfficientNetV2: Smaller Models and Faster Training."
  * [4] Ganaie, M. A., et al. (2022). "Ensemble deep learning: A review."
