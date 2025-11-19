import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Cargar el modelo preentrenado
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')  # Usa 'yolo11n.pt' para la versión más reciente

model = load_model()

st.title("Detección de Objetos con YOLO")
uploaded_file = st.file_uploader("Sube una imagen...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen original', use_column_width=True)

    # Convertir a numpy array para YOLO
    image_np = np.array(image)

    # Realizar la detección
    results = model(image_np)

    # Mostrar los resultados (plot() devuelve una imagen anotada)
    annotated_image = results[0].plot()
    st.image(annotated_image, caption='Imagen con detecciones', use_column_width=True)
