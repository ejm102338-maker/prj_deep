import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

class_name = ["마동석","장원영","카리나"]

@st.cache_resource
def load_model():
    model = models.resnet34(pretrained=False)
    num_fc_input = model.fc.in_features
    model.fc = nn.Linear(num_fc_input,3)
    model.load_state_dict(torch.load("models/best_model.pth",map_location='cpu'))
    model.eval()
    return model

# 모델 불러오기
model = load_model()

def transform_image(image):
    transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    return transform(image).unsqueeze(0)  #(1,3,224,224)

#메인화면 영역
st.title("연예인 분류기 (마동석,장원역,카리나)")

uploaded_file = st.file_uploader("이미지를 업로드해주세요",type=['jpg','jpeg','png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image,caption="업로드 이미지", use_container_width=False)
    model = load_model()
    input_tensor_image = transform_image(image)

    with torch.no_grad():
        outputs = model(input_tensor_image)
        _, preds = torch.max(outputs,1)
        pred_class = class_name[preds.item()]
        confidence = torch.softmax(outputs , dim=1)[0][preds.item()].item() * 100

    st.success(f'예측결과 : **{pred_class}**  ({confidence:.2f}% 확신)')