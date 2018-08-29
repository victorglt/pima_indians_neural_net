from keras.models import load_model
import numpy as np
from flask import jsonify
import tensorflow as tf
from flask import Flask, request

app = Flask(__name__)
model = None

global graph
graph = tf.get_default_graph()

def load_model():
    global model
    model = load_model('diabetes_model.h5')

@app.route("/diabetes/predict", methods=[ 'POST' ])
def get_diabetes_prediction():
    with graph.as_default():
        json = request.get_json()

        data = np.matrix([[
            json.get('pregnancies', 0),
            json.get('plasma_glucose', 0),
            json.get('diastolic_blood_pressure', 0),
            json.get('triceps_thickness', 0),
            json.get('serum_insulin',0),
            json.get('bmi', 0),
            json.get('diabetes_pedigree'),
            json.get('age',0)]])

        print(data)

        prediction = model.predict(data)

        return jsonify(prediction.tolist())

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print("* Loading Keras model and Flask starting server...please wait until server has fully started")
    load_model()
    app.run()