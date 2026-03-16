from fastapi import FastAPI,  UploadFile, File, HTTPException
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import json
import uuid
import os
import shutil
from PIL import Image
import io

app = FastAPI()
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@app.on_event('startup')
def load_model():
    print('========== 모델 불러오기 시작 ==============')
    global model
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512,3)
    
    checkpoint = torch.load('best_model.pth',map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else : 
        model.load_state_dict(checkpoint)

    print(device)
    model.to(device)
    model.eval()
    print('========== 모델 불러오기 끝!!! ==============')

transform_test = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]
)


@app.get('/')
def root():
    return {"result" : "Hi!!!"}


@app.post('/infer')
async def infer(file:UploadFile = File(...)):    #body : form-data

    allowed_ext = ["jpg","jpeg","png","webp"]
    ext = file.filename.split(".")[-1].lower()       #"a.png" -> split    ['a','png']

    if ext not in allowed_ext:
        return {'error':'이미지파일만 업로드하세요!!!!!'}
    
    newfile_name = f'{uuid.uuid4()}.{ext}'
    file_path = os.path.join('upload_img',newfile_name)

    with open(file_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    #-----------------추론코드--------------------------

    #img = await file.read()
    filepath = os.path.join('upload_img',newfile_name)
    img_data = Image.open(filepath).convert('RGB')
    print(img_data.size)
    input_tensor = transform_test(img_data).unsqueeze(0).to(device)  # (3,224,224) -> (1,3,224,224)

    pred = model(input_tensor)
    result = torch.argmax(pred,dim=1).item()    #0,1,2

    model_class = ['마동석','카리나','장원영']

    return {"result" : model_class[result] , "index" : result, "filename":newfile_name }