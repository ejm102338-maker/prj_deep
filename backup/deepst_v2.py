import streamlit as st
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

class_name = ["마동석","장원영","카리나"]

@st.cache_resource
def load_model():
    model = models.resnet34(pretrained = False)
    num_fc_input = model.fc.in_features
    model.fc = nn.Linear(num_fc_input,3) # 가중치 업데이트
    model.load_state_dict(torch.load('best_model.pth',map_location='cpu')) # 모델 불러오기
    model.eval()
    return model # 반환 st.cache_resource 설정으로 메모리에 한번 불러올때 저장

# 모델 불러오기
model = load_model()

# 전처리
def transform_image(image):
    transform=transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]
    )
    return transform(image).unsqueeze(0) # (1:배치사이즈,3:컬러,h:224,w:224)

# 메인화면영역
st.set_page_config(page_title="연예인 분류기(마동석,장원영,카리나)", layout="wide")
#st.title("연예인 분류기(마동석,장원영,카리나)")

uploaded_file = st.file_uploader("이미지를 첨부하세요!",type=['jpg','jpeg','png','webp'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB") # 흑백이미지인경우 채널수가 안맞아 일치시킴
    st.image(image,caption="업로드 이미지",use_container_width=True)

    model = load_model()

    input_tensor_image = transform_image(image)
    with torch.no_grad():
        outputs = model(input_tensor_image)
        _,preds = torch.max(outputs,1)
        pred_class = class_name[preds.item()]
        confidence = torch.softmax(outputs,dim=1)[0][preds.item()] * 100
        #torch.softmax(outputs,dim=1)[0] 예시: 0.8(마동석),0.1(장원영),0.25(카리나) 중 가장 확률이 큰값
    
    st.success(f"예측결과 : **{pred_class}** ({confidence:.2f}% 확신)") #소숫점 둘째자리까지 표현

    #streamlit run deepst_v2.py --server.port 6003