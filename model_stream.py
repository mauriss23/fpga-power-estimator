import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.preprocessing import StandardScaler

model=pickle.load(open("vlsi.pkl","rb"))

sc_x = pickle.load(open("sc_x.pkl", "rb"))
sc_y = pickle.load(open("sc_y.pkl", "rb"))

import os

import streamlit as st

bg_color = "lightblue"  

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {bg_color}; }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("FPGA based Design Power consumption Prediction")
st.write("This model predicts the power consumption")

LUTs=st.number_input("Enter LUTs used:")
#FFs,DSPs,IOs,Bit_Width,Pipeline,Unroll,Delay(ns)
FFs=st.number_input("Enter flip flops used:")
DSPs=st.number_input("Enter DSPs used:")
IOs=st.number_input("Enter the IOs used:")
Bit_Width=st.number_input("Enter the Bit_width:")
Pipeline=st.number_input("Enter pipelines if used:")
Unroll=st.number_input("Enter the number of Unrolls:")
Delay=st.number_input("Enter the delay the nanoseconds")



if st.button("Predict"):
    features=np.array([[LUTs,FFs,DSPs,IOs,Bit_Width,Pipeline,Unroll,Delay]],dtype=float)
    features_scaled=sc_x.transform(features)
    prediction_scaled=model.predict(features_scaled)
    prediction_real=sc_y.inverse_transform(prediction_scaled.reshape(1,-1))
    
    st.success(f"Predicted Power:",prediction_real[0],"(w)")

img=Image.open(r"C:\Users\Dell\Documents\extraction\fpga.jpg")
st.image(img, use_container_width=True)