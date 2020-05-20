#Front end(HTML,JavaScript),web service as well as the app,backend(python)
#Flask host web service and we also use it to host web app
from flask import Flask,render_template
import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
import traceback
from skimage.transform import resize, rescale, rotate, setup, warp, AffineTransform
from io import BytesIO
from flask_cors import CORS
from skimage.io import imsave
import tensorflow as tf

app1=Flask(__name__,static_url_path='/static')#Creating an instance,flask knows where to look for the templates and static files
CORS(app1)
graph = None

def get_model():
    global model, graph
    model=load_model('H:\my_model.h5')
    model.load_weights('H:\my_model.h5')
    graph = tf.get_default_graph()
    print("Model loaded....")

def preprocess_image(image,size):
    
    if image.mode != "RGB":
        image=image.convert("RGB")
    image=np.array(image)
    image1=resize(image,size)
    imsave('output.jpg', image1)
    image2=img_to_array(image1)#Convert the image into a numpy array
    image3=np.expand_dims(image2,axis=0)#Expands the dimensions
    image4=np.array(image3,dtype="float")/255.0
    return image4
print(" loading model.....")
get_model()



@app1.route('/project/p',methods=['GET','POST'])
#The endpoint name
def predict():
    message=request.get_json(force=True)
    encoded=message["image"]
    decoded=base64.b64decode(encoded.split(',')[1])
    decoded=np.asarray(decoded)
    image=Image.open(io.BytesIO(decoded))
    process_im=preprocess_image(image,size=(224,224))
    with graph.as_default():
        prediction=model.predict(process_im).tolist()#returns numpy array which is converted to a list
    

    response={'prediction':{
        'India Gate':prediction[0][0]*100,
        'Qutub Minar':prediction[0][1]*100,
        'Taj Mahal':prediction[0][2]*100#Since we are predicting on one image
              }
              
        }
    
    return jsonify(response) #Send it as json through the web back to the client

#in order to run your flask
#set FLASK_APP=backend.py
#flask run --host=0.0.0.0 (making it publicly visible)
#Run this on administrator in the command prompt
