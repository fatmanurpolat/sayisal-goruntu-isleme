import streamlit as st
import numpy as np
from PIL import Image, ImageFilter

def normalize_image(image):
    """Normalize image data to the range [0.0, 1.0]."""
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)

def erosion(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return morphological_operation(image, kernel, operation='erosion')

def dilation(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return morphological_operation(image, kernel, operation='dilation')

def opening(image, kernel_size=5):
    eroded = erosion(image, kernel_size)
    return dilation(eroded, kernel_size)

def closing(image, kernel_size=5):
    dilated = dilation(image, kernel_size)
    return erosion(dilated, kernel_size)

def morphological_operation(image, kernel, operation):
    pad_height, pad_width = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
    output = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                region = padded_image[y:y + kernel.shape[0], x:x + kernel.shape[1], c]
                if operation == 'erosion':
                    output[y, x, c] = np.min(region[kernel == 1])
                elif operation == 'dilation':
                    output[y, x, c] = np.max(region[kernel == 1])
    return output

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

    if st.session_state.menu_secimi == "Morfolojik İşlemler":
        selected_operation = st.sidebar.selectbox("İşlem Seçin", ["Aşındırma", "Genleşme", "Açma", "Kapama"])
        kernel_size = st.sidebar.number_input("Kernel Boyutu", min_value=3, max_value=21, step=2, value=5)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    if yuklenen_resim is not None:
        resim = Image.open(yuklenen_resim).convert('RGB')
        st.image(resim, caption="Yüklenen Resim", use_column_width=True)

        resim_dizisi = np.array(resim)

        if st.session_state.menu_secimi == "Morfolojik İşlemler":
            if selected_operation == "Aşındırma":
                islenmis_resim = erosion(resim_dizisi, kernel)
                islenmis_resim = normalize_image(islenmis_resim)
                st.image(islenmis_resim, caption="Aşındırma", use_column_width=True)

            elif selected_operation == "Genleşme":
                islenmis_resim = dilation(resim_dizisi, kernel)
                islenmis_resim = normalize_image(islenmis_resim)
                st.image(islenmis_resim, caption="Genleşme", use_column_width=True)

            elif selected_operation == "Açma":
                acilmis_resim = opening(resim_dizisi, kernel)
                acilmis_resim = normalize_image(acilmis_resim)
                st.image(acilmis_resim, caption="Açma", use_column_width=True)

            elif selected_operation == "Kapama":
                kapanmis_resim = closing(resim_dizisi, kernel)
                kapanmis_resim = normalize_image(kapanmis_resim)
                st.image(kapanmis_resim, caption="Kapama", use_column_width=True)


