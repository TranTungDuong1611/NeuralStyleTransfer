import streamlit as st
import numpy as np
import cv2
from NeuralStyleTransfer import *

st.title('Neural Style Transfer')
uploaded_content = st.file_uploader('Upload An Image For Content...', type=['jpg', 'jpeg', 'png'])

if uploaded_content is not None:
    file_content = np.asarray(bytearray(uploaded_content.read()), dtype=np.uint8)
    content_image = cv2.imdecode(file_content, 1)
    content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
    st.image(content_image, caption='Content Image', use_column_width=True)

uploaded_style = st.file_uploader('Upload An Image For Style...', type=['jpg', 'jpeg', 'png'])

if uploaded_style is not None:
    file_style = np.asarray(bytearray(uploaded_style.read()), dtype=np.uint8)
    style_image = cv2.imdecode(file_style, 1)
    style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
    st.image(style_image, caption='Style Image', use_column_width=True)
    
stepsizes = st.slider('Step Sizes', 500, 2000, 1)

if st.button('Start Transfer'):
    st.write('Transfering...')

    style_transfer = NeuralStyleTransfer()
    styled_image = style_transfer(content_image, style_image, stepsizes)
    st.image(styled_image, caption='Styled Image', use_column_width=True)