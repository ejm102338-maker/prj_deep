# uv pip install grad-cam
from fastapi import FastAPI, UploadFile,File, HTTPException
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import json
import os
import shutil

# uv add uuid 업로드된 파일 고유한 아이디 만들어주는 라이브러리
import uuid

app = FastAPI()

@app.get('/') # 주소 분기(라우터) (uvicorn main:app --reload --port 6003    ) 가상환경 상태에서 띄움 / Ctrl + C : 서버종료
def root():
    return {"result" : "Hi !!!"}

@app.post('/infer')
def infer(file:UploadFile = File(...)): # body : form-data > input data
    allowed_ext = ["jpg","jpeg","png","webp"]
    ext = file.filename.split(".")[-1].lower() # a.png
    
    if ext not in allowed_ext:
        return {"error":"이미지 파일만 업로드 하세요!"}

    newfile_name = f'{uuid.uuid4()}.{ext}' # 고유한 파일이름 재 네이밍

    # 파일 저장
    file_path = os.path.join('upload_img',newfile_name) # 저장할 경로

    with open(file_path,'wb') as buffer:
        shutil.copyfileobj(file.file,buffer)

    #--------------------------------------------------------------
    # 추론하는 코드
    #--------------------------------------------------------------

    return {"result":"카리나","index" : "2","filename" : newfile_name}