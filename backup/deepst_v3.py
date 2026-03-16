import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- 1. 환경 설정 및 모델 로드 ---
st.set_page_config(page_title="연예인 분류기", page_icon="👤", layout="centered")

# 디바이스 설정 (GPU가 있으면 사용, 없으면 CPU)
device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 클래스 이름 정의 (반드시 학습할 때 사용한 순서와 동일해야 합니다)
# 예시: 폴더 순서가 madongseok, karina, jangwonyoung 이었다면 그에 맞게 수정
CLASS_NAMES = ['마동석', '장원영', '카리나'] 
NUM_CLASSES = len(CLASS_NAMES)

# 모델 인스턴스 생성 및 가중치 로드 (캐싱을 통해 속도 향상)
@st.cache_resource
def load_model():
    # 학습할 때 사용한 것과 동일한 구조의 ResNet34 모델 생성
    model = models.resnet34(weights=None) # 학습된 가중치는 나중에 로드하므로 None
    
    # 마지막 레이어를 3개 클래스(마동석, 장원영, 카리나)로 변경
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # 학습된 가중치 파일(`.pth`) 로드
    # 파일명은 실제 가지고 계신 파일명으로 수정하세요.
    model_path = 'best_model.pth' 
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval() # 추론 모드로 설정
        return model
    else:
        st.error(f"모델 파일 '{model_path}'을 찾을 수 없습니다. 경로를 확인해 주세요.")
        return None

model = load_model()

# --- 2. 이미지 전처리 정의 ---
# 학습할 때 사용한 전처리(크기 조정, 정규화)와 동일하게 설정해야 합니다.
preprocess = transforms.Compose([
    transforms.Resize(256),            # 이미지 크기 조정
    transforms.CenterCrop(224),       # 중앙 크롭
    transforms.ToTensor(),             # Tensor로 변환 (0~1)
    transforms.Normalize(             # ImageNet 정규화 값 (학습 시 사용)
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- 3. 메인 화면 구성 ---
st.title("👤 초간단 연예인 분류기")
st.write("마동석, 장원영, 카리나 사진을 업로드하면 누구인지 분류해 드립니다!")

# 이미지 업로더 구성
uploaded_file = st.file_uploader("이미지 선택...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. 업로드된 이미지 표시
    image = Image.open(uploaded_file).convert('RGB') # RGBA 이미지를 RGB로 변환
    st.image(image, caption='업로드된 이미지', use_column_width=True)
    
    st.write("")
    
    if model is not None:
        with st.spinner('모델이 이미지를 분석 중입니다... 잠시만 기다려 주세요.'):
            # 2. 이미지 전처리
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0) # 배치 차원 추가 (1, 3, 224, 224)
            input_batch = input_batch.to(device)
            
            # 3. 모델 추론
            with torch.no_grad(): # 그래디언트 계산 비활성화
                output = model(input_batch)
                
            # 4. 결과 해석
            probabilities = torch.nn.functional.softmax(output[0], dim=0) # 확률로 변환
            top_prob, top_class_idx = torch.topk(probabilities, 1) # 가장 높은 확률의 클래스 찾기
            
            result_class = CLASS_NAMES[top_class_idx[0].item()]
            result_prob = top_prob[0].item() * 100 # 퍼센트로 변환
            
            # 5. 결과 출력
            st.success(f"분석 완료!")
            st.metric(label="예측 결과", value=result_class)
            st.write(f"**확률:** {result_prob:.2f}%")
            
            # 6. (선택사항) 모든 클래스의 확률 표시
            with st.expander("모든 클래스 확률 보기"):
                for i, prob in enumerate(probabilities):
                    st.write(f"{CLASS_NAMES[i]}: {prob.item()*100:.2f}%")

    else:
        st.warning("모델이 로드되지 않아 예측을 수행할 수 없습니다.")

    #streamlit run deepst_v3.py --server.port 6003