from sklearn import datasets
from joblib import load
import numpy as np
import json

#load the model

my_model = load('regressor_model.pkl')


#iris_data = datasets.load_iris()
#class_names = iris_data.target_names

def my_prediction(id):
    dummy = np.array(id)
    dummyT = dummy.reshape(1,-1)
    prediction = my_model.predict(dummyT)
    prediction = str(prediction)
    name_str = json.dumps(prediction)
    stringg = [name_str]
    return stringg