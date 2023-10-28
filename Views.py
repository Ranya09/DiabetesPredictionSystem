from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    data = pd.read_excel(r'C:/Users/Rania/Desktop/diabetes.xlsx')
    X = data.drop("Outcome", axis=1)
    Y = data['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    model = LogisticRegression()
    model.fit(x_train, y_train)

    val1 = float(request.GET.get('n1'))
    val2 = float(request.GET.get('n2'))
    val3 = float(request.GET.get('n3'))
    val4 = float(request.GET.get('n4'))
    val5 = float(request.GET.get('n5'))
    val6 = float(request.GET.get('n6'))
    val7 = float(request.GET.get('n7'))
    val8 = float(request.GET.get('n8'))

    # Prédiction
    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    if pred == 1:
        result1 = "Positive"
    else:
        result1 = "Negative"

    return render(request, 'Predict.html', {'result2':result1})
