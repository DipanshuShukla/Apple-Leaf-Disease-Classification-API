from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
from starlette.requests import Request
from tensorflow.keras.models import load_model
import numpy as np
from typing import List
import cv2,os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

classes_file = open('classes.txt','r')
classes = classes_file.read().split('\n')
classes_file.close()

def classify(contents):
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite('static/temp.png', cv2.resize(img,(256,256)))

    model = load_model(os.path.join('models', f'MobileNetV2_mini.h5'))
    resize = tf.image.resize(img, (256,256))
    yhat = model.predict(np.expand_dims(resize/255, 0))

    result = {x:"{0:.10f}".format(y*100) for x,y in zip(classes,yhat[0])}

    return {'result': result}

@app.get('/', response_class=HTMLResponse)
async def UI_endpoint(request: Request):
    return templates.TemplateResponse('ui_layout.html',{'request': request})

@app.post('/', response_class=HTMLResponse)
async def UI_endpoint(request: Request,imgFile: UploadFile = File(...)):
    print(imgFile.filename)
    contents = await imgFile.read()
    return templates.TemplateResponse('ui_layout.html',{'request': request, 'result' :classify(contents)['result']})

@app.post('/classify')
async def classification_endpoint(imgFile: UploadFile):
    contents = await imgFile.read()
    return classify(contents)