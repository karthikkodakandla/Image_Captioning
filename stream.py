# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:21:07 2020

@author: Vandhana
"""

import streamlit as st
import os
from PIL import Image
from app import predict
import tensorflow as tf
from PIL import Image
import requests
import base64
from io import BytesIO

main_bg = "background.jpg"
main_bg_ext = "jpg"


st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

st.sidebar.info("This is an Image Captioning web deployment Model.The application identifies the objects in \
                the picture and generates Caption. It was built using a Convolution Neural Network (CNN) for object identification and RNN to generate captions using sequence to sequence model (LSTM)")

st.set_option('deprecation.showfileUploaderEncoding', False)

#st.title("Image Captioning")
st.markdown("<h1 style='text-align: center; color: green;'>Image Captioning</h1>", unsafe_allow_html=True)
st.write("")

#def file_selector(folder_path='.'):
#    filenames = os.listdir(folder_path)
#    selected_filename = st.selectbox('Select a file', filenames)
#    return os.path.join(folder_path, selected_filename)
#
#
#if __name__ == '__main__':
#    # Select a file
#    if st.checkbox('Select a file in current directory'):
#        folder_path = '.'
#        if st.checkbox('Change directory'):
#            folder_path = st.text_input('Enter folder path', '.')
#        filename = file_selector(folder_path=folder_path)
#        st.write('You selected `%s`' % filename)
#        image = Image.open(filename)
#        st.image(image, caption='Selected Image.', use_column_width=True)
#        st.write("")
#        st.write("Just a second...")
#        label = predict('bike.jpg')
#        st.write("Caption for above image is : ",label)
        
#st.write()

def load_img(path_to_img):
  img = tf.io.read_file(path_to_img)
  #img = tf.image.decode_image(img, channels=3)
  #img = tf.image.resize(img, (224,224))
  return img
st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
status = st.radio("Hello, Do you want to Upload an Image or Insert an Image URL?",("Upload Image","Insert URL"))
if status == 'Upload Image':
    st.success("Please Upload an Image")
    file_up = st.file_uploader("Upload an image", type="jpg")
    if file_up is not None:
            image = Image.open(file_up)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            image = image.resize((224,224))
            st.write("Just a second...")
            label = predict(image)
            st.write("Caption for above image is : ",label)
            audio_file = open("text.mp3","rb").read()
            st.audio(audio_file,format='audio/mp3')
  
else:
    st.success("Please Insert Web URL")
    url = st.text_input("Insert URL below")
    if url:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        image = image.resize((224,224))
        st.write("Just a second...")
        label = predict(image)
        st.write("Caption for above image is : ",label)
        audio_file = open("text.mp3","rb").read()
        st.audio(audio_file,format='audio/mp3')
        
