import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
import os
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
from sklearn.datasets import load_files
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from tensorflow.keras.utils import plot_model  # Updated import
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.preprocessing import image
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from keras.models import load_model
from PIL import Image
from keras_preprocessing.image import load_img, img_to_array
import numpy as np

model_path = #r'./distracted-25-1.00.hdf5' #bring the h5 file from the notebook and ive its path
model = load_model(model_path)
#load the model
app = Flask(__name__)

def get_className(classNo):
	if classNo==0:
		return "Normal"
	elif classNo==1:
		return "Pneumonia"

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = load_img(img_path, target_size=(128, 128))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def getResult(img):

    image_data = paths_to_tensor([img]).astype('float32')/255 - 0.5
    predictions = model.predict(image_data)
    # Interpret predictions (example for classification)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_class_index = predicted_class[0]
    # print('Predicted Class Index:', predicted_class_index)
    # Extract the predicted class index
    # predicted_class_name = [name for name, idx in labels_id.items() if idx == predicted_class_index][0]
    # print('Predicted Class:', predicted_class_name)
    return predicted_class_index


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/mobile')
def mobile():
    return render_template('main.html')

maping = #your Converting into numerical values #{0: 'c8', 1: 'c4', 2: 'c1', 3: 'c9', 4: 'c5', 5: 'c2', 6: 'c6', 7: 'c3', 8: 'c0', 9: 'c7'}
    

class_name = dict()
class_name["c0"] = "SAFE_DRIVING"
class_name["c1"] = "TEXTING_RIGHT"
class_name["c2"] = "TALKING_PHONE_RIGHT"
class_name["c3"] = "TEXTING_LEFT"
class_name["c4"] = "TALKING_PHONE_LEFT"
class_name["c5"] = "OPERATING_RADIO"
class_name["c6"] = "DRINKING"
class_name["c7"] = "REACHING_BEHIND"
class_name["c8"] = "HAIR_AND_MAKEUP"
class_name["c9"] = "TALKING_TO_PASSENGER"
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        # result=get_className(value) 
        # return result
        class_value = maping[value]
        res = class_name[class_value]

        return res
    return None


if __name__ == '__main__':
    app.run(debug=True)