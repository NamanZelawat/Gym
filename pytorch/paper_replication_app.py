import streamlit as st
from PIL import Image

st.title("Vision Transformer Paper Replication")

st.write("This is a replication of the ViT paper using PyTorch.")

st.write("## Upload an image")
st.write("Upload an image to classify from pizza, steak, or sushi. Only 3 categories are supported.")
uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    st.write("## Prediction")
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=250)
    st.write("## Result")
    st.write(f"The image is a pizza, steak, or sushi.")


