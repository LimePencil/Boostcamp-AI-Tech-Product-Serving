import streamlit as st
import yaml
from PIL import Image
import io
from predict import load_model,predict_image



st.title("Mask Classification Model")

with open("streamlit_prototyping_special_mission_1/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


model=load_model(config['model_path'],config['hparam_path'])

st.markdown("<h6 style='text-align: right'> Model: "+str(type(model).__name__)+" </h6>",unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png"])

if uploaded_file:
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    columns=st.columns(3)
    with columns[1]:
        st.image(image, caption='Uploaded Image',use_column_width=True)
    y_hat = predict_image(model, image_bytes)
    label = config['classes'][y_hat.item()]
    col1,col2,col3=st.columns(3)
    with col1:
        st.metric("Gender",label[1])
    with col2:
        st.metric("Age",label[2])
    with col3:
        st.metric("Mask",label[0])
st.markdown("<h6 style='text-align: right; color: grey'> Created by Jaeyoung Shin </h6>",unsafe_allow_html=True)