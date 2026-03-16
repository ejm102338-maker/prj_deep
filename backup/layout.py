import streamlit as st
st.set_page_config(page_title="모델 대시보드", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.header("필터")
    dataset = st.selectbox("데이터셋", ["Val","Test"])
    smooth = st.slider("스무딩", 1, 25, 5)
    show_points = st.checkbox("포인트 표시", False)

# --- Header ---
st.title("모델 성능 대시보드")
st.caption("실험 비교 · 지표 요약 · 예측 분포")

# --- KPI Row ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Best Val Acc", "93.2%")
k2.metric("Min Val Loss", "0.183")
k3.metric("Latency(ms)", "12.4")
k4.metric("Params(M)", "21.8")

# --- Tabs ---
t1, t2, t3 = st.tabs(["학습 곡선", "지표표/혼동행렬", "예측 샘플"])

with t1:
    c1, c2 = st.columns([3,2])
    with c1:
        st.subheader("Loss Curve")
        st.line_chart({"train":[0.9,0.6,0.4,0.25], "val":[1.0,0.7,0.5,0.3]})
    with c2:
        st.subheader("Accuracy Curve")
        st.line_chart({"train":[0.5,0.7,0.85,0.92], "val":[0.45,0.65,0.8,0.9]})

with t2:
    st.subheader("지표 테이블")
    st.table({"precision":[0.91,0.88], "recall":[0.90,0.86], "f1":[0.905,0.87]})
    with st.expander("혼동행렬 보기"):
        st.dataframe({"Pred 0":[88,5], "Pred 1":[7,100]})

with t3:
    left, right = st.columns([2,3])
    with left:
        st.subheader("입력 샘플")
        st.image("https://placehold.co/300x300", caption="업로드/샘플")
    with right:
        st.subheader("Top-k 확률")
        st.bar_chart({"A":[0.7], "B":[0.2], "C":[0.1]})