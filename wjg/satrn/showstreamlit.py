import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
from ocr_core import ocr_core
from matplotlib import pyplot as plt
from inference import main


st.title('[신경망과 딥러닝] 수학수식 인식')

# 업로드 기능
st.subheader("수식 업로드")
uploaded_file = st.file_uploader("이미지를 선택하세요...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='업로드된 이미지', use_column_width=True)
    
    if st.button('Predict'):
        with st.spinner('잠시만 기다려주세요...'):
            image = Image.open(uploaded_file)
            test_x = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
            test_x = Image.fromarray(test_x)
            
            val = ocr_core(test_x)
        
        st.latex(val)
        st.write(f'LaTeX : {val}')
else:
    st.warning("업로드된 이미지를 선택해주세요.")