import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
import pickle

from flask import Flask, request, render_template
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
col=['Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'Pclass', 'Embarked']
@app.route('/')
def loadPage():
    return render_template('index.html')
@app.route('/predict', methods=['POST','GET'])
def ServivalPrediction():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features, dtype=float)]
    prediction= model.predict(final)
    if prediction==1:
        o1="Passenger is survived"
    else:
        o1="Passenger is not survived"

    return render_template('index.html',pred=o1)
if __name__=='__main__':
    app.run(debug=True, port= 2020)