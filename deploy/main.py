# ::: Import modules and packages :::
# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Import Keras dependencies
from keras.models import model_from_json
from tensorflow.python.framework import ops
from keras.preprocessing import image

# Import other dependecies
import numpy as np
import os
# import sys


# Path to our saved model
# sys.path.append(os.path.abspath("./model"))
ops.reset_default_graph()

# ::: Flask App Engine :::
# Define a Flask app
app = Flask(__name__)

# ::: Prepare Keras Model :::
# Model files
MODEL_ARCHITECTURE = './model/model_vgg16_2020-06-22_12-52.json'
MODEL_WEIGHTS = './model/model_weight_2020-06-22_12-52.h5'

# Load the model from external files
json_file = open(MODEL_ARCHITECTURE, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Get weights into the model
model.load_weights(MODEL_WEIGHTS)
print('Model loaded. Check http://127.0.0.1:5000/')


# ::: MODEL FUNCTIONS :::
def model_predict(img_path, model):
    new_img = image.load_img(img_path, target_size=(224, 224))
    print(type(new_img))
    img = image.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    print(type(img), img.shape)

    print("Following is our prediction:")
    prediction = model.predict(img)
    return prediction


# ::: FLASK ROUTES
@app.route('/', methods=['GET'])
def index():
    # Main Page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():

    # Constants:
    li = [
        'Apple___Apple_scab', 
        'Apple___Black_rot', 
        'Apple___Cedar_apple_rust', 
        'Apple___healthy', 
        'Blueberry___healthy', 
        'Cherry_(including_sour)___Powdery_mildew', 
        'Cherry_(including_sour)___healthy', 
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
        'Corn_(maize)___Common_rust_', 
        'Corn_(maize)___Northern_Leaf_Blight', 
        'Corn_(maize)___healthy', 
        'Grape___Black_rot', 
        'Grape___Esca_(Black_Measles)', 
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
        'Grape___healthy', 
        'Orange___Haunglongbing_(Citrus_greening)', 
        'Peach___Bacterial_spot', 
        'Peach___healthy', 
        'Pepper,_bell___Bacterial_spot', 
        'Pepper,_bell___healthy', 
        'Potato___Early_blight', 
        'Potato___Late_blight', 
        'Potato___healthy', 
        'Raspberry___healthy', 
        'Soybean___healthy', 
        'Squash___Powdery_mildew', 
        'Strawberry___Leaf_scorch', 
        'Strawberry___healthy', 
        'Tomato___Bacterial_spot', 
        'Tomato___Early_blight', 
        'Tomato___Late_blight', 
        'Tomato___Leaf_Mold', 
        'Tomato___Septoria_leaf_spot', 
        'Tomato___Spider_mites Two-spotted_spider_mite', 
        'Tomato___Target_Spot', 
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
        'Tomato___Tomato_mosaic_virus', 
        'Tomato___healthy'
        ]
    
    if request.method=='POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make a prediction
        prediction = model_predict(file_path, model)

        # decode the results into a list of tuples (class, description, probability)
        # (one such list for each sample in the batch)
        d = prediction.flatten()
        j = d.max()
        for index,item in enumerate(d):
            if item == j:
                predicted_class = li[index]
        print('Our prediction is {}.'.format(predicted_class.lower()))

        return str(predicted_class).lower()

if __name__ == '__main__':
	app.run(debug = True)