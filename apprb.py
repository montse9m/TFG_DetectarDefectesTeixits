import os
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input

st.set_page_config(page_title="Classificador de Defectes", layout="centered")

MODEL_PATH = 'model_inceptionv3_50.h5'
IMG_SIZE = (299, 299)
THRESHOLD = 0.3

RESULTATS_DIR = "resultats"
DEFECTE_DIR = os.path.join(RESULTATS_DIR, "defecte")
NODEFECTE_DIR = os.path.join(RESULTATS_DIR, "nodefecte")
os.makedirs(DEFECTE_DIR, exist_ok=True)
os.makedirs(NODEFECTE_DIR, exist_ok=True)


@st.cache_resource
def carregar_model():
    return load_model(MODEL_PATH)

model = carregar_model()


def preprocessa_img(img: np.ndarray):
    img = cv2.resize(img, IMG_SIZE)
    img = preprocess_input(np.expand_dims(img.astype(np.float32), axis=0))
    return img

def classifica_i_guarda(img_np, original_img, prefix="captura"):
    pred = model.predict(img_np)[0][0]
    classe = "defecte" if pred > THRESHOLD else "nodefecte"
    carpeta = DEFECTE_DIR if classe == "defecte" else NODEFECTE_DIR
    nom_fitxer = f"{prefix}_{classe}_{int(time.time())}.jpg"
    ruta = os.path.join(carpeta, nom_fitxer)
    cv2.imwrite(ruta, original_img)
    return classe, pred, ruta


st.title("Classificador de defectes")

mode = st.radio("Selecciona el mode:", ("Adjuntar imatge", "Capturar amb càmera"))

if mode == "Adjuntar imatge":
    fitxer = st.file_uploader("Puja una imatge", type=["jpg", "jpeg", "png"])
    if fitxer is not None:
        img_pil = Image.open(fitxer).convert("RGB")
        img_np = np.array(img_pil)
        img_model = preprocessa_img(img_np)
        classe, conf, path = classifica_i_guarda(img_model, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), "manual")

        st.image(img_pil, caption="Imatge original", use_container_width=True)
        st.success(f"Predicció: **{classe.upper()}** ({conf:.2f})")
        st.caption(f"Guardada a: `{path}`")

elif mode == "Capturar amb càmera":
    start = st.button("Iniciar captura cada 5 segons")

    if start:
        cam = cv2.VideoCapture(1)  

        if not cam.isOpened():
            st.error("No s'ha pogut accedir a la càmera.")
        else:
            frame_placeholder = st.empty()
            while True:
                ret, frame = cam.read()
                if not ret or frame is None:
                    st.error("Error en capturar la imatge.")
                    break

                img_model = preprocessa_img(frame)
                classe, conf, path = classifica_i_guarda(img_model, frame, "camera")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, caption=f"Predicció: {classe.upper()} ({conf:.2f})", channels="RGB")
                time.sleep(5)
