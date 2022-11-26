from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2,os

app = FastAPI()

classes_file = open('classes.txt','r')
classes = classes_file.read().split('\n')
print(classes)
classes_file.close()

@app.get('/', response_class=HTMLResponse)
async def UI_endpoint():
    with open('templates/index.html') as f:
        return f.read() 


@app.post('/classify')
async def classification_endpoint(imgFile: UploadFile):
    contents = await imgFile.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    model = load_model(os.path.join('models', f'MobileNetV2_mini.h5'))
    resize = tf.image.resize(img, (256,256))
    yhat = model.predict(np.expand_dims(resize/255, 0))

    result = {x:"{0:.10f}".format(y*100) for x,y in zip(classes,yhat[0])}

    return {'input_file_name':imgFile.filename, 'input_img_shape': str(img.shape), 'result': result}