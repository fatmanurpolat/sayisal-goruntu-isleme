import numpy as np
import matplotlib.pyplot as plt
import streamlit as st 
from PIL import Image
import cv2

def compute_histogram(image):
    if len(image.shape) == 3:  # Renkli resim
        histogram = [np.histogram(image[:, :, i], bins=256, range=(0, 256))[0] for i in range(3)]
    else:  # Gri tonlamalı resim
        histogram = np.histogram(image, bins=256, range=(0, 256))[0]
    return histogram

def plot_histogram(histogram):
    plt.figure()
    if isinstance(histogram, list):
        # Renkli resim histogramı
        colors = ('b', 'g', 'r')
        color_labels = ('Mavi kanal', 'Yeşil kanal', 'Kırmızı kanal')
        for hist, color, label in zip(histogram, colors, color_labels):
            plt.plot(hist, color=color, label=label)
        plt.xlim([0, 256])
        plt.xlabel('Piksel Yoğunluğu')
        plt.ylabel('Frekans')
        plt.title('Renkli Histogram')
        plt.legend()  # Her renk kanalını ayırt etmek için açıklama ekleyin
    else:
        # Gri tonlamalı resim histogramı
        plt.plot(histogram, color='k', label='Gri tonlama')
        plt.xlim([0, 256])
        plt.xlabel('Piksel Yoğunluğu')
        plt.ylabel('Frekans')
        plt.title('Gri Tonlama Histogramı')
        plt.legend()  # Gri tonlamayı ayırt etmek için açıklama ekleyin
    st.pyplot(plt)

def histogram_equalization(image):
    if len(image.shape) == 3:  # Renkli resim
        channels = [cv2.equalizeHist(image[:, :, i]) for i in range(3)]
        equalized_image = cv2.merge(channels)
    else:  # Gri tonlamalı resim
        equalized_image = cv2.equalizeHist(image)
    return equalized_image
