import streamlit as st

import pickle
import numpy as np
from sklearn.linear_model import Perceptron

model = pickle.load(open('per_model-66130701722.sav','rb'))

st.title("Iris species Prediction using Preceptron")

x1=st.slider('select Input1',0.0,10.0,3.0)
x2=st.slider('select Input1',0.0,10.0,5.0)
x3=st.slider('select Input1',0.0,10.0,4.0)
x4=st.slider('select Input1',0.0,10.0,7.0)

xnew = np.array([[6,3,5,3]])#.reshape(1,-1)

xnew

pred = model.predict(xnew)
st.write("## Prediction resulte:")
st.write('Species:')
