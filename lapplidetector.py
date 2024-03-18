import streamlit as st
import numpy as np
from ultralytics import YOLO
import base64
import cv2
from PIL import Image
import sys
from utils import set_background

set_background('AdobeStock_225120346_Preview.jpeg')

st.write('<p style="font-size:70px; color:#060606; background-color:#f1f2f6; padding:3px;">Image Object Detection</p>', unsafe_allow_html=True)
st.write('<p style="font-size:40px; color:#060606; background-color:#f1f2f6; padding:3px;">Please upload an image</p>', unsafe_allow_html=True)

# Interface Streamlit
image = st.file_uploader("Upload an image in the following formats: png, jpg, jpeg", type=[".png", ".jpg", ".jpeg"])

if image is not None:

    model = YOLO("yolov8s.pt")
    print("-----------")
    # Utiliser le modèle pour prédire sur l'image téléchargée
    results = model(source=Image.open(image), save=True)
  # Récupérer la sortie imprimée
    
    annote=results[0].verbose()
    
    st.write(f'<p style="font-size:40px; color:#060606; background-color:#f1f2f6; padding:1px;">Here is what we detected in your image:<br>{annote}</p>', unsafe_allow_html=True)
    st.image(results[0].plot())

    # Instructions pour sauvegarder l'image sur le disque
    result_file = 'result.jpg'
    results[0].save(filename=result_file)  # Sauvegarder l'image résultante

    # the download button
    st.download_button(
        label="Download the image with the detections",
        data=open(result_file, 'rb').read(),
        file_name=result_file,
        mime='image/jpeg'
    )



