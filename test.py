import cv2
import streamlit as st

# Initialize the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    st.error("Error: Could not open camera.")
else:
    st.success("Camera is working.")
    ret, frame = camera.read()
    if ret:
        # Convert the frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the image using Streamlit
        st.image(frame, channels="RGB", caption="Test Frame")
    else:
        st.error("Error: Could not read frame.")

# Release the camera
camera.release()
