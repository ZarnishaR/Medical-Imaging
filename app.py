import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import detection.detect as detect
import classification.classify as classify
import segmentation.segment as segment


def train_models():
    detect.train()
    print("[INFO] Training Detection model done!")
    classify.train()
    print("[INFO] Training Classification model done!")
    
    segment.prepare_input()    
    segment.train()
    print("[INFO] Training Segmentation model done!")
    
    
def main():
    
    st.sidebar.title("Settings")
    st.sidebar.subheader("Parameters")
    
    app_mode = st.sidebar.selectbox('Choose the App Mode', ['About App', 'Object Detection', 'Object Classification', "Object Segmentation"])
    
    
    if app_mode == 'About App':
        
        st.header("Medical Imaging")
        st.write("YOLOv8-based application for classification, detection, and segmentation tasks in medical imaging.")
        st.write("Select a mode from the sidebar to get started.")
        
    elif app_mode == "Object Detection":
        
        st.header("Object Detection")
        
        st.sidebar.markdown("----")
        confidence = st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, value=0.35)
        
        img_file_buffer_detect = st.sidebar.file_uploader("Upload an image", type=['jpg','jpeg', 'png'], key=0)
        DEMO_IMAGE = "DEMO_IMAGES/BloodImage_00000_jpg.rf.5fb00ac1228969a39cee7cd6678ee704.jpg"
        
        if img_file_buffer_detect is not None:
            img = cv.imdecode(np.fromstring(img_file_buffer_detect.read(), np.uint8), 1)
            image = np.array(Image.open(img_file_buffer_detect))
        else:
            img = cv.imread(DEMO_IMAGE)
            image = np.array(Image.open(DEMO_IMAGE))
        st.sidebar.text("Original Image")
        st.sidebar.image(image)
        
        detect.predict(img, confidence, st)
        
    elif app_mode == "Object Classification":
        
        st.header("Classification")
        
        st.sidebar.markdown("----")
        
        img_file_buffer_classify = st.sidebar.file_uploader("Upload an image", type=['jpg','jpeg', 'png'], key=1)
        DEMO_IMAGE = "DEMO_IMAGES/094.png"
        
        if img_file_buffer_classify is not None:
            img = cv.imdecode(np.fromstring(img_file_buffer_classify.read(), np.uint8), 1)
            image = np.array(Image.open(img_file_buffer_classify))
        else:
            img = cv.imread(DEMO_IMAGE) 
            image = np.array(Image.open(DEMO_IMAGE))
        st.sidebar.text("Original Image")
        st.sidebar.image(image)
        
        classify.predict(img, st)
        
    elif app_mode == "Object Segmentation":
        
        st.header("Segmentation")
        
        st.sidebar.markdown("----")
        
        img_file_buffer_segment = st.sidebar.file_uploader("Upload an image", type=['jpg','jpeg', 'png'], key=2)
        DEMO_IMAGE = "DEMO_IMAGES/benign (2).png"
        
        if img_file_buffer_segment is not None:
            img = cv.imdecode(np.fromstring(img_file_buffer_segment.read(), np.uint8), 1)
            image = np.array(Image.open(img_file_buffer_segment))
        else:
            img = cv.imread(DEMO_IMAGE)
            image = np.array(Image.open(DEMO_IMAGE))
        st.sidebar.text("Original Image")
        st.sidebar.image(image)
        
        segment.predict(img, st)


if __name__ == "__main__":
    try:
        # train_models()
        main()
    except SystemExit:
        pass