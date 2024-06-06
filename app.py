
import streamlit as st
from PIL import Image
import numpy as np
from edge_detection import *
from blurring import *
from histogram import *
from morphological_operations import *
import cv2

# Custom CSS to change the font and colors
st.markdown(
    """
    <style>
    body {
        font-family: 'Comic Sans MS', cursive, sans-serif;
        background-color: #F5F5F5; /* Light background */
    }
    .stButton>button {
        background-color: #FFB6C1; /* Light pink */
        color: black;
        border-radius: 16px;
        height: 50px;
        width: 100%;
        box-shadow: 2px 2px 5px grey;
        font-size: 18px;
        margin-bottom: 10px;
    }
    .stFileUploader>div>button {
        background-color: #FFB6C1; /* Light pink */
        color: black;
        border-radius: 16px;
        height: 50px;
        width: 100%;
        box-shadow: 2px 2px 5px grey;
        font-size: 18px;
    }
    .stSidebar .stButton>button {
        background-color: #FFB6C1; /* Light pink */
        color: black;
        border-radius: 16px;
        height: 50px;
        width: 100%;
        box-shadow: 2px 2px 5px grey;
        font-size: 18px;
        margin-bottom: 10px;
    }
    .stSubheader {
        color: #FF69B4; /* Hot pink */
    }
    </style>
    """,
    unsafe_allow_html=True
)

def normalize_image(image):
    """Normalize image data to the range [0.0, 1.0]."""
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)

def main():
    st.title("Görüntü İşleme")

    # Initialize session state
    if 'menu_secimi' not in st.session_state:
        st.session_state.menu_secimi = None

    st.sidebar.subheader("İşlem Seçin")
    if st.sidebar.button("Kenar Tespiti"):
        st.session_state.menu_secimi = "Kenar Tespiti"
    if st.sidebar.button("Bulanıklaştırma"):
        st.session_state.menu_secimi = "Bulanıklaştırma"
    if st.sidebar.button("Histogram"):
        st.session_state.menu_secimi = "Histogram"
    if st.sidebar.button("Morfolojik İşlemler"):
        st.session_state.menu_secimi = "Morfolojik İşlemler"

    yuklenen_resim = st.file_uploader("Bir resim yükleyin", type=["jpg", "jpeg", "png"])

    if yuklenen_resim is not None:
        resim = Image.open(yuklenen_resim).convert('RGB')
        st.image(resim, caption="Yüklenen Resim", use_column_width=True)

        resim_dizisi = np.array(resim)
        
        # Kenar Tespiti
        if st.session_state.menu_secimi == "Kenar Tespiti":
            st.subheader("Kenar Tespiti Sonuçları")

            st.write("1. Sobel Operatörü")
            st.write("2. Canny Kenar Algılama")

            if st.button("Sobel Operatörünü Uygula"):
                sobel_resim = sobel_edge_detection(resim_dizisi)
                sobel_resim = normalize_image(sobel_resim)
                st.image(sobel_resim, caption="Sobel Kenar Tespiti", use_column_width=True)

            if st.button("Canny Kenar Algılamayı Uygula"):
                canny_resim = canny_edge_detection(resim_dizisi)
                canny_resim = normalize_image(canny_resim)
                st.image(canny_resim, caption="Canny Kenar Tespiti", use_column_width=True)
        
        # Bulanıklaştırma
        elif st.session_state.menu_secimi == "Bulanıklaştırma":
            st.subheader("Bulanıklaştırma Sonuçları")

            st.write("1. Ortalama Bulanıklaştırma")
            st.write("2. Medyan Bulanıklaştırma")
            st.write("3. Gaussian Bulanıklaştırma")

            if st.button("Ortalama Bulanıklaştırmayı Uygula"):
                bulanık_resim = average_blur(resim_dizisi)
                bulanık_resim = normalize_image(bulanık_resim)
                st.image(bulanık_resim, caption="Ortalama Bulanıklaştırma", use_column_width=True)

            if st.button("Medyan Bulanıklaştırmayı Uygula"):
                bulanık_resim = median_blur(resim_dizisi)
                bulanık_resim = normalize_image(bulanık_resim)
                st.image(bulanık_resim, caption="Medyan Bulanıklaştırma", use_column_width=True)

            if st.button("Gaussian Bulanıklaştırmayı Uygula"):
                bulanık_resim = gaussian_blur(resim_dizisi)
                bulanık_resim = normalize_image(bulanık_resim)
                st.image(bulanık_resim, caption="Gaussian Bulanıklaştırma", use_column_width=True)
                
        # Histogram
        elif st.session_state.menu_secimi == "Histogram":
            st.subheader("Histogram")

            histogram = compute_histogram(resim_dizisi)
            plot_histogram(histogram)

            if st.button("Histogram Eşitlemeyi Uygula"):
                esitleme_resim = histogram_equalization(resim_dizisi)
                esitleme_resim = normalize_image(esitleme_resim)
                st.image(esitleme_resim, caption="Histogram Eşitleme", use_column_width=True)

        # Morfolojik İşlemler
        elif st.session_state.menu_secimi == "Morfolojik İşlemler":
            st.subheader("Morfolojik İşlemler")

            st.write("1. Aşındırma")
            st.write("2. Genleşme")
            st.write("3. Açma")
            st.write("4. Kapama")

            if st.button("Aşındırmayı Uygula"):
                islenmis_resim = erosion(resim_dizisi)
                islenmis_resim = normalize_image(islenmis_resim)
                st.image(islenmis_resim, caption="Aşındırma", use_column_width=True)

            if st.button("Genleşmeyi Uygula"):
                islenmis_resim = dilation(resim_dizisi)
                islenmis_resim = normalize_image(islenmis_resim)
                st.image(islenmis_resim, caption="Genleşme", use_column_width=True)

if __name__ == "__main__":
    main()