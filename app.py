import numpy as np
import streamlit as st
from PIL import Image
import cv2
from edge_detection import sobel_edge_detection, canny_edge_detection
from blurring import average_blur, median_blur, gaussian_blur
from histogram import compute_histogram, plot_histogram, histogram_equalization
from morphological_operations import erosion, dilation, opening, closing
from region_growing import region_growing

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
    if st.sidebar.button("Region Growing"):
        st.session_state.menu_secimi = "Region Growing"

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

            threshold = st.slider("Histogram Eşik Değeri", min_value=0, max_value=255, value=128)

            if st.button("Histogram Eşitlemeyi Uygula"):
                esitleme_resim = histogram_equalization(resim_dizisi, threshold)
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

            if st.button("Açmayı Uygula"):
                acilmis_resim = opening(resim_dizisi)
                acilmis_resim = normalize_image(acilmis_resim)
                st.image(acilmis_resim, caption="Açma", use_column_width=True)

            if st.button("Kapamayı Uygula"):
                kapanmis_resim = closing(resim_dizisi)
                kapanmis_resim = normalize_image(kapanmis_resim)
                st.image(kapanmis_resim, caption="Kapama", use_column_width=True)

        # Region Growing
        elif st.session_state.menu_secimi == "Region Growing":
            st.subheader("Region Growing")

            seed_x = st.number_input("Başlangıç Noktası X Koordinatı", min_value=0, max_value=resim_dizisi.shape[0]-1, value=0)
            seed_y = st.number_input("Başlangıç Noktası Y Koordinatı", min_value=0, max_value=resim_dizisi.shape[1]-1, value=0)
            threshold = st.slider("Eşik Değeri", min_value=0, max_value=255, value=15)

            if st.button("Region Growing Uygula"):
                seed_point = (seed_x, seed_y)
                gray_image = cv2.cvtColor(resim_dizisi, cv2.COLOR_RGB2GRAY)  # Gri tonlamalı resme çeviriyoruz
                region_growing_resim = region_growing(gray_image, seed_point, threshold)
                region_growing_resim = normalize_image(region_growing_resim)
                st.image(region_growing_resim, caption="Region Growing", use_column_width=True)

if __name__ == "__main__":
    main()
