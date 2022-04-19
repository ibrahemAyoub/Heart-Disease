# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template
from id3_algorithm import *
df = pd.read_csv('heart_disease_male.csv', header=None)
col_names = ['age', 'chest_pain_type', 'rest_blood_pressure', 'blood_sugar', 'rest_electro', 'max_heart_rate', 'exercice_angina','disease']
df.columns = col_names

df = df[~df['age'].isin(['?'])]
df = df[~df['chest_pain_type'].isin(['?'])]
df = df[~df['rest_blood_pressure'].isin(['?'])]
df = df[~df['blood_sugar'].isin(['?'])]
df = df[~df['rest_electro'].isin(['?'])]
df = df[~df['max_heart_rate'].isin(['?'])]
df = df[~df['exercice_angina'].isin(['?'])]
df = df[~df['disease'].isin(['?'])]

from sklearn.model_selection import train_test_split
#split the data into train and test set
train,test = train_test_split(df, test_size=0.30, random_state=0)

tree = id3(train, 'disease')

# Create application
app = Flask(__name__)

# Bind home function to URL
@app.route('/')
def home():
    return render_template('Heart Disease Classifier.html')

# Bind predict function to URL
@app.route('/predict', methods =['POST'])
def pred():

    # Predict features
    prediction =predict(tree,{"age":request.form["age"],"chest_pain_type":request.form["chest_pain_type"],"rest_blood_pressure":request.form["rest_blood_pressure"],"blood_sugar":request.form["blood_sugar"],
    "rest_electro":request.form["rest_electro"],"max_heart_rate":request.form["max_heart_rate"],"exercice_angina":request.form["exercice_angina"]})
    
    output =prediction
    print(output)
    
    # Check the output values and retrive the result with html tag based on the value
    if output == 'negative':
        return render_template('Heart Disease Classifier.html', 
                               result = 'Negative, The patient is not likely to have heart disease!')
    else:
        return render_template('Heart Disease Classifier.html', 
                               result = 'Positive, The patient is likely to have heart disease!')

if __name__ == '__main__':
#Run the application
    #app.debug = True
    app.run(host="0.0.0.0",port=5000)
    
    