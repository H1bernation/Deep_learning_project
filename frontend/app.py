import streamlit as st
import plotly.graph_objects as go
from PIL import Image
import requests
import time
import base64
import io

# ==================== 페이지 설정 ====================
st.set_page_config(
    page_title="CROLO : AI 스킨케어 서비스",
    page_icon="🧴",
    layout="wide"
)

# ==================== 세션 상태 초기화 ====================
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'chatbot_open' not in st.session_state:
    st.session_state.chatbot_open = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}

# ==================== API 호출 함수 ====================
def validate_image_client(image_file):
    """
    클라이언트 측 이미지 검증
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # 파일 크기 체크
        image_file.seek(0, 2)
        file_size = image_file.tell()
        image_file.seek(0)
        
        if file_size == 0:
            return False, "빈 파일입니다. 이미지를 선택해주세요."
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            size_mb = file_size / (1024 * 1024)
            return False, f"파일 크기는 10MB 이하여야 합니다. (현재: {size_mb:.1f}MB)"
        
        # 실제 이미지인지 확인
        try:
            image_bytes = image_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()
            
            # 이미지 크기 확인
            width, height = image.size
            if width < 100 or height < 100:
                return False, f"이미지 해상도가 너무 낮습니다. (현재: {width}x{height}, 최소: 100x100)"
            
        except Exception as e:
            return False, "유효하지 않은 이미지 파일입니다."
        
        return True, None
        
    except Exception as e:
        return False, f"파일 검증 중 오류가 발생했습니다: {str(e)}"

def call_backend_api(image_file, user_profile):
    """백엔드 API 호출 - 예외처리 강화"""
    try:
        # 클라이언트 측 검증
        is_valid, error_msg = validate_image_client(image_file)
        if not is_valid:
            st.error(f"⚠️ {error_msg}")
            return None
        
        # API 호출
        image_bytes = image_file.getvalue()
        url = "http://skin-analysis-backend:8000/api/v1/analyze-and-recommend"
        
        files = {
            'file': ('image.jpg', image_bytes, 'image/jpeg')
        }
        
        data = {
            'skin_type': user_profile['skin_type'],
            'budget_min': user_profile['budget_min'],
            'budget_max': user_profile['budget_max']
        }
        
        response = requests.post(url, files=files, data=data, timeout=60)
        
        # 상태 코드별 처리
        if response.status_code == 400:
            # 클라이언트 오류 (잘못된 요청)
            try:
                error_detail = response.json().get('detail', '잘못된 요청입니다.')
                st.error(f"⚠️ {error_detail}")
            except:
                st.error(f"⚠️ 요청 처리 중 오류가 발생했습니다.")
            return None
            
        elif response.status_code == 500:
            # 서버 오류
            st.error("❌ 서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
            return None
        
        response.raise_for_status()
        
        result = response.json()
        
        # 백엔드 응답 검증
        if not result:
            st.error("❌ 서버로부터 응답을 받지 못했습니다.")
            return None
        
        # success 필드 체크 
        if 'analysis' in result:
            analysis = result['analysis']
            if isinstance(analysis, dict) and not analysis.get('success', True):
                error_msg = analysis.get('error_message', '분석에 실패했습니다.')
                st.error(f"⚠️ {error_msg}")
                return None
        
        return result
        
    except requests.exceptions.Timeout:
        st.error("⏱️ 요청 시간이 초과되었습니다. 이미지 크기를 줄이거나 다시 시도해주세요.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 서버에 연결할 수 없습니다. 네트워크 연결을 확인해주세요.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API 오류: {e.response.status_code}")
        try:
            error_detail = e.response.json()
            if 'detail' in error_detail:
                st.error(f"상세 오류: {error_detail['detail']}")
        except:
            pass
        return None
    except Exception as e:
        st.error(f"❌ 예상치 못한 오류: {str(e)}")
        return None

def call_chatbot_api(question, analysis, recommendations, user_profile):
    """RAG 챗봇 API 호출 - 예외처리 강화"""
    try:
        # 입력 검증
        if not question or len(question.strip()) == 0:
            return "질문을 입력해주세요."
        
        if len(question) > 500:
            return "질문이 너무 깁니다. 500자 이내로 입력해주세요."
        
        url = "http://skin-analysis-backend:8000/api/v1/chat"
        
        payload = {
            "question": question.strip(),
            "analysis": analysis['symptoms'],
            "recommendations": recommendations['products'][:10],
            "user_preferences": user_profile
        }
        
        response = requests.post(url, json=payload, timeout=30)
        
        # 상태 코드 체크
        if response.status_code == 400:
            return "⚠️ 잘못된 질문입니다. 다시 입력해주세요."
        elif response.status_code == 500:
            return "❌ 서버 오류가 발생했습니다. 다시 시도해주세요."
        
        response.raise_for_status()
        
        result = response.json()
        
        if 'answer' not in result:
            return "❌ 답변을 받지 못했습니다. 다시 시도해주세요."
        
        return result['answer']
        
    except requests.exceptions.Timeout:
        return "⏱️ 응답 시간이 초과되었습니다. 다시 시도해주세요."
    except requests.exceptions.ConnectionError:
        return "🔌 서버에 연결할 수 없습니다. 네트워크를 확인해주세요."
    except Exception as e:
        return f"일시적인 오류가 발생했습니다. 다시 시도해주세요."

def load_css(css_path):
    """외부 CSS 파일 로드"""
    try:
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # CSS 파일이 없어도 계속 진행 (에러 표시 안 함)
        pass
    except Exception as e:
        # 기타 오류도 조용히 무시
        pass

# ==================== 1. 랜딩 페이지 ====================
def landing_page():
    load_css("styles.css")
    
    col1, col2 = st.columns([1.15, 1])
    
    with col1:
        try:
            with open('/app/contents/skin_ai_main_page.mp4', 'rb') as video_file:
                video_bytes = video_file.read()
                video_base64 = base64.b64encode(video_bytes).decode()
            
            st.markdown(f"""
            <div class="image-container">
                <video class="background-video" autoplay loop muted playsinline>
                    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                </video>
            </div>
            """, unsafe_allow_html=True)
        except:
            # 비디오 로드 실패 시 대체 화면
            st.markdown("""
            <div class="image-container">
                <div>
                    <div style="text-align: center; color: white;">
                        <div style="font-size: 100px;">👤</div>
                        <p style="font-size: 20px; margin-top: 20px;">AI 피부 분석</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="right-content">', unsafe_allow_html=True)
        
        st.markdown("""
        <div>
            <div class="logo-text">CROLO</div>
            <div class="tagline">AI 피부 분석 기반 스킨케어 추천 서비스</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("START", key="start_btn"):
            st.session_state.page = 'input'
            st.rerun()
        
        st.markdown("""
        <div class="feature-grid">
            <div class="feature-box">
                <img src="https://cdn-icons-png.flaticon.com/512/2922/2922671.png" class="icon" alt="얼굴"/>
                <p>얼굴<br>사진 분석</p>
            </div>
            <div class="feature-box">
                <img src="https://cdn-icons-png.flaticon.com/512/1048/1048948.png" class="icon" alt="AI"/>
                <p>AI<br>증상 분류</p>
            </div>
            <div class="feature-box">
                <img src="https://cdn-icons-png.flaticon.com/512/1875/1875676.png" class="icon" alt="제품"/>
                <p>맞춤형<br>제품 추천</p>
            </div>
            <div class="feature-box">
                <img src="https://cdn-icons-png.flaticon.com/512/134/134914.png" class="icon" alt="챗봇"/>
                <p>챗봇<br>상담</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


# ==================== 3. 분석 페이지 ====================
def input_page():
    # 1. 기본 스타일 로드
    load_css("styles.css")
    
    st.markdown("""
        <style>
            /* [1] 상단 공백 제거 */
            .main .block-container {
                padding-top: 1rem !important;
                padding-bottom: 2rem !important;
                max-width: 100% !important;
            }
            header[data-testid="stHeader"] { display: none !important; }

            /* [2] 전체 배경: 아주 연한 회색 (박스와 대비) */
            [data-testid="stAppViewContainer"] {
                background: #f7f9fc !important;
            }
            
            /* [3] 컬럼(박스) 디자인: 내용물을 감싸는 흰색 박스 */
            /* 화면에 보이는 메인 컬럼 2개만 타겟팅 */
            [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                background-color: #ffffff !important;  /* 흰색 배경 */
                border: 1px solid #e0e0e0 !important;  /* 은은한 회색 테두리 */
                border-radius: 15px !important;        /* 둥근 모서리 */
                padding: 25px !important;              /* 안쪽 여백 */
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important; /* 그림자 */
                height: fit-content !important;        /* [핵심] 높이를 내용물에 맞춤 */
            }

            /* [4] 텍스트 색상 정리 */
            h1, h2, h3, p, span, div, label {
                color: #333 !important;
                font-family: 'MaruBuri', sans-serif;
                text-shadow: none !important;
            }
            
            /* [5] 타이틀 그라데이션 */
            h1.main-title {
                background: linear-gradient(to right, #008c99, #00eaff) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
                font-family: 'Aritaburi' !important;
            }

            /* [6] 라벨 폰트 스타일 */
            .stSelectbox label, .stSlider label, [data-testid="stRadio"] label p {
                font-family: 'Aritaburi' !important;
                font-size: 18px !important;
                font-weight: 700 !important;
                color: #111 !important;
                margin-bottom: 8px;
            }
            
            /* [7] 라디오 버튼 박스 */
            [data-testid="stRadio"] > div {
                background: #f8f9fa !important;
                padding: 15px;
                border-radius: 10px;
                border: 1px solid #eee;
            }

            /* [8] 셀렉트 박스 */
            div[data-baseweb="select"] > div {
                background-color: #ffffff !important;
                border: 1px solid #ccc !important;
                border-radius: 10px !important;
                color: #000 !important;
                font-weight: 600 !important;
            }
            div[data-baseweb="select"] span { color: #000 !important; }
            
            /* [9] 섹션 헤더 디자인 */
            .section-header {
                font-family: 'Aritaburi';
                font-size: 22px;
                font-weight: 700 !important;
                color: #111 !important;
                border-left: 5px solid #333;
                padding-left: 12px;
                margin-bottom: 20px;
                margin-top: 0px;
            }
            
            /* [10] 파일 업로더 */
            [data-testid="stFileUploader"] section {
                background-color: #fcfcfc !important;
                border: 1px dashed #ccc !important;
            }
            [data-testid="stFileUploader"] button {
                border-color: #008c99 !important;
                color: #008c99 !important;
            }
            
            /* [11] 분석 버튼 */
            div.stButton > button[kind="primary"] {
                width: 100% !important;
                margin-top: 10px !important;
                background: linear-gradient(90deg, #008c99 0%, #00eaff 100%) !important;
                border: none !important;
                border-radius: 12px !important;
                font-family: 'Aritaburi' !important;
                font-size: 20px !important;
                font-weight: 700 !important;
                padding: 15px 0 !important;
                box-shadow: 0 4px 15px rgba(0, 234, 255, 0.3) !important;
            }
            div.stButton > button[kind="primary"] p { color: white !important; }
            div.stButton > button[kind="primary"]:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 25px rgba(0, 234, 255, 0.5) !important;
            }
            
            /* 슬라이더 */
            .stSlider [data-testid="stThumb"] { background-color: #008c99 !important; }
            .stSlider [data-testid="stTickBar"] { background-color: #ddd !important; }
        </style>
    """, unsafe_allow_html=True)
    
    # 3. 페이지 타이틀
    st.markdown("""
        <div style="
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            justify-content: center; 
            width: 100%;
            margin-top: 0px; 
            margin-bottom: 40px;
        ">
            <h1 class="main-title" style="
                font-size: 50px; 
                font-weight: 700; 
                margin: 0; 
                line-height: 1.2;
                text-align: center;
            ">
                Skin Analysis Setup
            </h1>
            <p style="
                font-family: 'MaruBuri'; 
                color: #555; 
                font-size: 18px; 
                font-weight: 600; 
                margin: 15px 0 0 0;
                text-align: center;
                letter-spacing: -0.5px;
            ">
                정확한 AI 분석을 위해 사진과 정보를 입력해주세요.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # 4. 레이아웃
    col1, col2 = st.columns([1, 1], gap="large")
    
    # [왼쪽 박스]
    with col1:
        st.markdown('<div class="section-header">📸 Photo Upload</div>', unsafe_allow_html=True)
        mode = st.radio("입력 방식 선택", ["카메라 촬영", "파일 업로드"], horizontal=True, label_visibility="collapsed")
        
        if mode == "파일 업로드":
            img = st.file_uploader("이미지를 업로드하세요", type=['jpg', 'png', 'jpeg'])
            if img:
                st.session_state.uploaded_image = img
                st.markdown('<div style="margin-top:20px; border-radius:10px; overflow:hidden; border:1px solid #00eaff;">', unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="height: 200px; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 2px dashed #bbb; border-radius: 10px; margin-top: 20px; background: #fcfcfc;">
                    <span style="font-size: 40px; opacity: 0.5;">📂</span>
                    <span style="font-size: 15px; color: #777; margin-top: 10px; font-weight: 600;">이미지를 드래그하거나 선택하세요</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            img = st.camera_input("카메라로 촬영하세요")
            if img:
                st.session_state.uploaded_image = img
    
    # [오른쪽 박스]
    with col2:
        st.markdown('<div class="section-header">👤 User Profile</div>', unsafe_allow_html=True)
        
        skin = st.selectbox("피부 타입 (Skin Type)", ["복합성", "지성", "건성", "민감성", "여드름성"])
        st.session_state.user_profile['skin_type'] = skin
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">💰 Budget Range</div>', unsafe_allow_html=True)
        budget = st.slider("화장품 구매 예산 (KRW)", 0, 70000, (20000, 30000), 5000, format="₩%d")
        st.session_state.user_profile['budget_min'] = budget[0]
        st.session_state.user_profile['budget_max'] = budget[1]
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # 분석 버튼
        if st.button("🔍 Analyze Now", use_container_width=True, type="primary"):
            if st.session_state.uploaded_image:
                st.session_state.page = 'result'
                st.rerun()
            else:
                st.warning("⚠️ 사진을 먼저 업로드해주세요.")

# ==================== 3. 결과 페이지 (Split Layout: Analysis + Chat) ====================

def result_page():
    # [스타일] 가로 배치 카드 + 카카오톡/토스 스타일 + 타이핑 애니메이션
    st.markdown("""
        <style>
            /* 1. 배경 및 기본 폰트 */
            [data-testid="stAppViewContainer"] { background: #ffffff !important; }
            h2, h3, p, span, div, label, li { color: #333333 !important; font-family: 'MaruBuri', sans-serif; }

            /* 2. 제목 그라데이션 */
            h1.result-title {
                font-family: 'Aritaburi' !important;
                background: linear-gradient(90deg, #008c99 0%, #00eaff 100%) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
                font-size: 42px !important;
                margin-bottom: 20px !important;
                text-align: center !important;
                width: 100% !important;
                display: block !important;
            }
            
            /* 3. 탭 디자인 */
            button[data-baseweb="tab"] { color: #888 !important; background: transparent !important; font-family: 'Aritaburi' !important; }
            button[data-baseweb="tab"][aria-selected="true"] { color: #008c99 !important; background: rgba(0, 140, 153, 0.1) !important; border: 1px solid #008c99 !important; }

            /* 4. 제품 카드 디자인 */
            a.product-link { text-decoration: none !important; color: inherit !important; display: block; }
            .product-card {
                display: flex; flex-direction: row; align-items: center; 
                background: #ffffff; border: 1px solid #eee; border-radius: 12px; 
                padding: 12px; margin-bottom: 15px; transition: transform 0.2s, box-shadow 0.2s;
            }
            a.product-link:hover .product-card {
                transform: translateY(-3px); box-shadow: 0 5px 15px rgba(0, 140, 153, 0.15); border-color: #008c99;
            }

            /* 5. 채팅 UI (보라색 & 프로필) */
            .user-container { display: flex; justify-content: flex-end; margin-bottom: 15px; }
            .user-bubble {
                background-color: #624bec !important; color: white !important;
                padding: 12px 16px !important; border-radius: 18px 2px 18px 18px !important;
                max-width: 80% !important; font-size: 14px !important; line-height: 1.5 !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important; white-space: pre-wrap !important;
                text-align: left !important; font-family: 'MaruBuri', sans-serif !important;
            }

            .bot-container { display: flex; align-items: flex-start; margin-bottom: 15px; }
            .bot-profile-icon {
                width: 38px; height: 38px; background-color: #eee; border-radius: 50%;
                display: flex; align-items: center; justify-content: center;
                font-size: 20px; margin-right: 10px; flex-shrink: 0;
            }
            .bot-content-wrapper { display: flex; flex-direction: column; max-width: 80%; }
            .bot-name { font-size: 12px; color: #888; margin-bottom: 4px; margin-left: 2px; font-family: 'Aritaburi', sans-serif !important; }
            .bot-bubble {
                background-color: #f2f4f6 !important; color: #333 !important;
                padding: 12px 16px !important; border-radius: 2px 18px 18px 18px !important;
                font-size: 14px !important; line-height: 1.6 !important;
                white-space: pre-wrap !important; box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
                font-family: 'MaruBuri', sans-serif !important;
            }

            /* 🌟 [NEW] 타이핑 애니메이션 (...) */
            .typing-indicator {
                display: flex; align-items: center; justify-content: center; height: 24px;
            }
            .typing-dot {
                width: 6px; height: 6px; margin: 0 3px; background-color: #888; border-radius: 50%;
                animation: typing 1.4s infinite ease-in-out both;
            }
            .typing-dot:nth-child(1) { animation-delay: -0.32s; }
            .typing-dot:nth-child(2) { animation-delay: -0.16s; }
            
            @keyframes typing {
                0%, 80%, 100% { transform: scale(0); }
                40% { transform: scale(1); }
            }
        </style>
    """, unsafe_allow_html=True)

    if st.session_state.analysis_result is None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("<h3 style='font-family:Aritaburi; color:#008c99;'>📸 Analyzing...</h3>", unsafe_allow_html=True)
            if st.session_state.uploaded_image:
                st.image(st.session_state.uploaded_image, use_container_width=True)
        with col2:
            video_path = "/app/contents/loading.mp4"
            
            try:
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                    encoded_video = base64.b64encode(video_bytes).decode()
                
                video_html = f"""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%;">
                    <video autoplay loop muted playsinline style="width: 80%; border-radius: 20px; box-shadow: 0 4px 20px rgba(0, 140, 153, 0.2);">
                        <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
                        이 브라우저는 동영상을 지원하지 않습니다.
                    </video>
                    <h2 style="color:#008c99; font-family:'Aritaburi'; margin-top: 20px;">AI Skin Analysis</h2>
                    <p style="color:#666;">피부 데이터를 정밀 분석 중입니다.</p>
                </div>
                """
                st.markdown(video_html, unsafe_allow_html=True)
                
            except FileNotFoundError:
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                st.markdown("""
                <div style="text-align: center;">
                    <div style="font-size: 60px; margin-bottom: 20px;">🔄</div>
                    <h2 style="color:#008c99; font-family:'Aritaburi';">AI Skin Analysis</h2>
                    <p style="color:#666;">피부 데이터를 정밀 분석 중입니다.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with st.spinner(''):
            result = call_backend_api(st.session_state.uploaded_image, st.session_state.user_profile)
            if result:
                st.session_state.analysis_result = result
                st.rerun() 
            else:
                if st.button("🔄 다시 시도"):
                    st.session_state.page = 'input'
                    st.session_state.analysis_result = None
                    st.rerun()
                if st.button("⬅️ 돌아가기"):
                    st.session_state.page = 'input'
                    st.session_state.analysis_result = None
                    st.rerun()

    else:
        main_col, chat_col = st.columns([1.6, 1], gap="large")
        result = st.session_state.analysis_result

        # ========== [왼쪽] 분석 결과 영역 ==========
        with main_col:
            st.markdown('<h1 class="result-title">AI Skin Analysis Report</h1>', unsafe_allow_html=True)
            
            # 피부 검출 & 차트
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            sub_c1, sub_c2 = st.columns([1, 1])
            with sub_c1:
                st.markdown("<h3 style='font-family:MaruBuri; color:#333; font-size:20px;'>📸 피부 검출 결과</h3>", unsafe_allow_html=True)
                if st.session_state.uploaded_image:
                    st.image(st.session_state.uploaded_image, use_container_width=True)
                    if 'analysis' in result and 'detection' in result['analysis']:
                        boxes = result['analysis']['detection']['boxes']
                        label_map = {'chin': '턱', 'forehead': '이마', 'left_cheek': '왼쪽 볼', 'right_cheek': '오른쪽 볼', 'nose': '코', 'left_eye': '왼쪽 눈', 'right_eye': '오른쪽 눈'}
                        tags = "".join([f"<span class='product-tag'>#{label_map.get(box['class'], box['class'])} </span>" for box in boxes])
                        st.markdown(f"<div style='margin-top:15px;'>{tags}</div>", unsafe_allow_html=True)
            with sub_c2:
                st.markdown("<h3 style='font-family:MaruBuri; color:#333; font-size:20px;'>📊 4대 지표 분석</h3>", unsafe_allow_html=True)
                symptoms = result['analysis']['symptoms']
                categories = ['주름', '모공', '색소', '탄력']
                values = [symptoms['wrinkle'], symptoms['pore'], symptoms['pigmentation'], symptoms['sagging']]
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself', fillcolor='rgba(0, 140, 153, 0.2)', line=dict(color='#008c99', width=2), marker=dict(size=4, color='#008c99')))
                fig.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=True, range=[0, 3], tickfont=dict(color='#666'), gridcolor='rgba(0,0,0,0.1)'), angularaxis=dict(tickfont=dict(color='#333', size=12), gridcolor='rgba(0,0,0,0.1)')), margin=dict(t=20, b=20, l=30, r=30), height=280, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # 제품 추천 리스트
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.markdown("<h3 style='font-family:MaruBuri; color:#333; font-size:20px;'>🧴 AI 추천 제품</h3>", unsafe_allow_html=True)
            
            products = result['recommendations']['products']
            tab_toner, tab_lotion, tab_serum = st.tabs(["토너", "로션", "세럼"])
            
            def display_products_grid(product_list):
                if not product_list:
                    st.markdown("<p style='text-align:center; color:#888; padding:20px;'>추천 제품이 없습니다.</p>", unsafe_allow_html=True)
                    return

                for p in product_list:
                    default_img = "https://images.unsplash.com/photo-1620916566398-39f1143ab7be?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&q=80"
                    link_url = p.get('url')  
                    img_src = p.get('image') if p.get('image') else default_img
                    if not link_url: link_url = "#"

                    try: rating_val = float(p['avg_rating'])
                    except: rating_val = 0.0
                    rating_str = f"{rating_val:.1f}"

                    card_html = f"""
<a href="{link_url}" target="_blank" class="product-link">
<div class="product-card">
    <div style="width: 100px; height: 100px; flex-shrink: 0; border-radius: 8px; overflow: hidden; margin-right: 15px; border: 1px solid #eee;">
        <img src="{img_src}" style="width: 100%; height: 100%; object-fit: cover; display:block;" onerror="this.src='{default_img}'">
    </div>
    <div style="flex-grow: 1;">
        <div style="font-size: 16px; font-weight: bold; color: #333; margin-bottom: 4px;">{p['name']}</div>
        <div style="font-size: 12px; color: #888; margin-bottom: 6px;">{p['brand']}</div>
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
            <span style="color: #008c99; font-weight: bold; font-size: 15px;">₩{p['price']:,}</span>
            <span style="color: #f1c40f; font-size: 13px;">★ {rating_str}</span>
        </div>
        <div style="font-size: 11px; color: #666; background: #f8f9fa; padding: 6px; border-radius: 6px; line-height: 1.4;">
            💡 {p['explanation']}
        </div>
    </div>
</div>
</a>
"""
                    st.markdown(card_html, unsafe_allow_html=True)

            with tab_toner: display_products_grid([p for p in products if '토너' in p['category']])
            with tab_lotion: display_products_grid([p for p in products if '로션' in p['category']])
            with tab_serum: display_products_grid([p for p in products if '세럼' in p['category'] or '에센스' in p['category']])
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("🔄 새로운 분석 시작"):
                st.session_state.page = 'input'
                st.session_state.analysis_result = None
                st.session_state.chat_history = []
                st.rerun()

        # ========== [오른쪽] 챗봇 영역 (타이핑 애니메이션 적용) ==========
        with chat_col:
            st.markdown("""
            <div class="chat-container-embedded">
                <div class="chat-header">
                    <div style="display:flex; align-items:center;">
                        <div class="bot-icon">🤖</div>
                        <span>CROLO 스킨 어시스턴트</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            chat_container = st.container(height=450)
            
            # [1] 채팅 히스토리 출력
            with chat_container:
                if not st.session_state.chat_history:
                    st.markdown("""
                    <div class="bot-container">
                        <div class="bot-profile-icon">🤖</div>
                        <div class="bot-content-wrapper">
                            <div class="bot-name">CROLO</div>
                            <div class="bot-bubble">안녕하세요! 피부 분석 결과에 대해 궁금한 점을 물어보세요.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                for chat in st.session_state.chat_history:
                    if chat['role'] == 'user':
                        st.markdown(f"""
                        <div class="user-container">
                            <div class="user-bubble">{chat['content']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="bot-container">
                            <div class="bot-profile-icon">🤖</div>
                            <div class="bot-content-wrapper">
                                <div class="bot-name">CROLO</div>
                                <div class="bot-bubble">{chat['content']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # [2] 입력창 및 처리 로직 (애니메이션 핵심 부분)
            st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
            with st.form(key='chat_form', clear_on_submit=True):
                user_input = st.text_input("질문", placeholder="예: 제 피부 타입에 더 잘 맞는 건 어떤 거예요?", label_visibility="collapsed")
                submit_button = st.form_submit_button("전송", use_container_width=True)
                
                if submit_button and user_input:
                    # 1. 유저 질문 세션에 추가
                    st.session_state.chat_history.append({'role': 'user', 'content': user_input})
                    
                    # 2. [즉시 렌더링] 유저 말풍선 수동으로 그리기
                    chat_container.markdown(f"""
                    <div class="user-container">
                        <div class="user-bubble">{user_input}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 3. [로딩 애니메이션] 봇 답변 자리에 '...' 애니메이션 표시
                    loading_placeholder = chat_container.empty()
                    loading_placeholder.markdown("""
                    <div class="bot-container">
                        <div class="bot-profile-icon">🤖</div>
                        <div class="bot-content-wrapper">
                            <div class="bot-name">CROLO</div>
                            <div class="bot-bubble">
                                <div class="typing-indicator">
                                    <div class="typing-dot"></div>
                                    <div class="typing-dot"></div>
                                    <div class="typing-dot"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 4. API 호출 (답변 생성 대기)
                    bot_response = call_chatbot_api(
                        question=user_input,
                        analysis=result['analysis'],
                        recommendations=result['recommendations'],
                        user_profile=st.session_state.user_profile
                    )
                    
                    # 5. 세션에 답변 추가 후 새로고침
                    st.session_state.chat_history.append({'role': 'bot', 'content': bot_response})
                    st.rerun()
            
            st.markdown('</div></div>', unsafe_allow_html=True)

# ==================== 메인 라우터 ====================
def main():
    if st.session_state.page == 'landing':
        landing_page()
    elif st.session_state.page == 'input':
        input_page()
    elif st.session_state.page == 'result':
        result_page()

if __name__ == "__main__":
    main()
