# uv add streamlit\n",
# uv run streamlit hello  #초기페이지 실행\n",
# 안되는경우(pyprject.toml 수정)\n",
# streamlit>=1.220\n",
# pandas>=0.25,<3\n",
# uv lock\n",
# uv sync"
# uv run streamlit run deepst.py :(실행하기)

import streamlit as st

# st.title("Hello Streamlit")
# st.write("첫번째 스트림릿 페이지 입니다.")

# name = st.text_input("이름을 입력해 주세요")

# if st.button("인사버튼"):
#     st.success(f'안녕하세요 {name}님! 반갑습니다.')
# 제목 / 텍스트 출력
st.title("스트림릿 제목")
st.header("헤더")
st.subheader("서브헤더")
st.text("일반 텍스트")
st.markdown("**마크다운 지원** :sparkles:")
st.code("print('Hello World')", language="python")

# 레이아웃
col1,col2,col3 = st.columns(3)
with col1:
    with st.container(border=True):
        st.write("왼쪽 컬럼")
col2.write("가운데 컬럼")
col3.write("오른쪽 컬럼")

with st.expander("펼치기/접기"):
    st.write("관리자 전화번호 : 010-1111-0000")

# 텍스트 입력
name = st.text_input("이름 입력")

# 숫자 입력 / 슬라이더
age = st.number_input("나이 입력", min_value=0, max_value=120, value=25)
score = st.slider("점수", 0, 100, 50)

# 체크박스 / 라디오 버튼 / 셀렉트박스
agree = st.checkbox("동의합니다")
option = st.radio("좋아하는 색상", ["빨강", "파랑", "초록"])
select = st.selectbox("과목 선택", ["수학", "과학", "영어"])
multi = st.multiselect("취미 선택", ["독서", "운동", "게임"])

# 버튼
if st.button("클릭"):
    st.success("버튼 눌림")

# 파일 업로드
uploaded_file = st.file_uploader("파일 업로드", type=["jpg","png","csv"])

# 미디어 출력
# 이미지
from PIL import Image
img = Image.open("./img/image_4.jpg")
st.image(img, caption="장원영", use_column_width=True)

if st.button("이미지 클릭"):
    st.image(uploaded_file, caption="업로드이미지", use_column_width=True)

# 오디오 / 비디오
# st.audio("music.mp3")
# st.video("video.mp4")

# 데이터 출력
import pandas as pd
df = pd.DataFrame({"이름":["철수","영희"], "점수":[90,80]})
st.table(df)       # 정적 테이블
st.dataframe(df)   # 인터랙티브 테이블

# 차트
import numpy as np
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a","b","c"])
st.line_chart(chart_data)
st.bar_chart(chart_data)
st.area_chart(chart_data)

# 상태 & 인터랙션
# 진행 상황 표시
import time
progress = st.progress(0)
for i in range(100):
    time.sleep(0.05)
    progress.progress(i+1)

# 메시지 알림
st.success("성공 메시지")
st.error("에러 메시지")
st.warning("경고 메시지")
st.info("정보 메시지")

# 요약

# - **입력 컴포넌트**: `text_input`, `slider`, `checkbox`, `selectbox`, `file_uploader`
# - **출력 컴포넌트**: `write`, `image`, `audio`, `video`, `dataframe`, `chart`
# - **레이아웃 & 상호작용**: `columns`, `expander`, `progress`, `status message`

# 👉 Streamlit은 “**데이터 → 입력 → 모델 → 출력**” 흐름을 자연스럽게 UI로 구성할 수 있도록 설계되어 있어, 딥러닝 모델 시연이나 대시보드 제작에 최적화되어 있습니다.