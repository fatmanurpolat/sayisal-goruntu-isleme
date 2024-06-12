import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
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

def histogram_equalization(image, threshold=128):
    if len(image.shape) == 3:  # Renkli resim
        channels = []
        for i in range(3):
            channel = image[:, :, i]
            channel = np.clip(channel, 0, threshold)
            equalized_channel = cv2.equalizeHist(channel)
            channels.append(equalized_channel)
        equalized_image = cv2.merge(channels)
    else:  # Gri tonlamalı resim
        image = np.clip(image, 0, threshold)
        equalized_image = cv2.equalizeHist(image)
    return equalized_image
