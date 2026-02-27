# uv pip install grad-cam
from fastapi import FastAPI, UploadFile,File, HTTPException
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import json
import os
import shutil
from PIL import Image
import io
from contextlib import asynccontextmanager

# uv add uuid 업로드된 파일 고유한 아이디 만들어주는 라이브러리
import uuid

# 방법2 로드될때 한번 실행
@asynccontextmanager
async def lifespan(app:FastAPI) :
    print('==== 모델 불러오기 시작 ====')
    # 가중치가 학습된 모델 불러오기
    global model # 전역변수로 사용
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512,3)
    checkpoint = torch.load('best_model.pth',map_location=device)

    if 'model_state_dic' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dic']) # 가중치가 있는경우 가중치 가져옴
    else :
        model.load_state_dict(checkpoint) # 가중치가 없는경우 전체

    model.eval() # 테스트 버전으로 전환
    model.to(device)
    print('==== 모델 불러오기 종료 ====')

    yield
    #서버 종료시 실행
    print("서버 종료!")

app = FastAPI(lifespan=lifespan) # 방법 2
# app = FastAPI()
device = 'cuda' if torch.cuda.is_available else 'cpu'
model = None

# 방법1 로드될때 한번 실행
# @app.on_event('startup')
# @AbstractAsyncContextManager
# def load_model() :
#     print('==== 모델 불러오기 시작 ====')
#     # 가중치가 학습된 모델 불러오기
#     global model # 전역변수로 사용
#     model = models.resnet34(pretrained=True)
#     model.fc = nn.Linear(512,3)
#     checkpoint = torch.load('best_model.pth',map_location=device)

#     if 'model_state_dic' in checkpoint:
#         model.load_state_dict(checkpoint['model_state_dic']) # 가중치가 있는경우 가중치 가져옴
#     else :
#         model.load_state_dict(checkpoint) # 가중치가 없는경우 전체

#     model.eval() # 테스트 버전으로 전환
#     model.to(device)
#     return model
#     print('==== 모델 불러오기 종료 ====')

# 데이터 전처리
transform_test = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]
)

@app.get('/') # 주소 분기(라우터) (uvicorn main:app --reload --port 6003    ) 가상환경 상태에서 띄움 / Ctrl + C : 서버종료
def root():
    return {"result" : "Hi !!!"}

@app.post('/infer')
async def infer(file:UploadFile = File(...)): # body : form-data > input data
    img = await file.read()
    allowed_ext = ["jpg","jpeg","png","webp"]
    ext = file.filename.split(".")[-1].lower() # a.png
    
    if ext not in allowed_ext:
        return {"error":"이미지 파일만 업로드 하세요!"}

    newfile_name = f'{uuid.uuid4()}.{ext}' # 고유한 파일이름 재 네이밍

    # 파일 저장
    file_path = os.path.join('upload_img',newfile_name) # 저장할 경로

    with open(file_path,'wb') as buffer:
        buffer.write(img)
        shutil.copyfileobj(file.file,buffer)

    #--------------------------------------------------------------
    # 추론하는 코드
    #--------------------------------------------------------------

    # 이미지 불러오기
    img_data = Image.open('./upload_img/'+newfile_name).convert('RGB')

    # 전처리
    tensor_data = transform_test(img_data).unsqueeze(0).to(device) # 맨앞(0번 인덱스)에 차원을 하나더 추가 한다

    #with torch.no_grad():
    pred = model(tensor_data) # 예측
    result = torch.argmax(pred,dim=1).item() # 가장큰수가 예측 결과값

    model_class = ['마동석','카리나','장원영']
    #--------------------------------------------------------------
    # //추론하는 코드
    #--------------------------------------------------------------


    return {"result":model_class[result],"index" : result,"filename" : newfile_name}