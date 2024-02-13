import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Perceptron

filename = 'per_model-66130701722.sav'
model = pickle.load(open(filename, 'rb'))

st.title('Iris Species Prediction Using Preception')

x1 = st.slider('select Input1',0.0,10.0,0.1)
x2 = st.slider('select Input2',0.0,10.0,0.1)
x3 = st.slider('select Input3',0.0,10.0,0.1)
x4 = st.slider('select Input4',0.0,10.0,0.1)

xnew = np.array([x1,x2,x3,x4]).reshape(1,-1)

predict = model.predict(xnew)

st.write("##Prediction Result")
st.write('Species',predict[0])