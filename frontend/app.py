import streamlit as st

st.set_page_config(
    page_title="피부 분석 AI",
    page_icon="🧴",
    layout="wide"
)

st.title("🧴 피부 분석 및 화장품 추천 AI")

st.markdown("""
## 환영합니다!

이 시스템은 딥러닝 기반으로 피부 상태를 분석하고 맞춤형 화장품을 추천합니다.

### 주요 기능:
- 📸 얼굴 사진 기반 피부 상태 분석
- 🤖 AI 기반 피부 증상 분류
- 💄 개인 맞춤형 제품 추천
- 💬 대화형 챗봇 상담

*현재는 개발 초기 단계입니다.*
""")

if st.button("서버 연결 테스트"):
    st.success("✅ Streamlit이 정상 작동 중입니다!")
    st.info("FastAPI 서버 연결은 나중에 구현됩니다.")