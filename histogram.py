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
    def equalize_channel(channel):
        # Histogram
        hist, bins = np.histogram(channel.flatten(), bins=256, range=[0,256])
        
        # Kümülatif dağılım fonksiyonu (CDF)
        cdf = hist.cumsum()
        
        # CDF'yi normalize etme
        cdf_normalized = cdf * float(hist.max()) / cdf.max()
        
        # CDF'nin min değerini al
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        
        equalized_channel = cdf[channel]
        return equalized_channel

    # Renkli resmi griye çevir
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if len(image.shape) == 3:  # Renkli resim
        channels = []
        for i in range(3):
            channel = gray_image[:, :]
            channel = np.clip(channel, 0, threshold)
            equalized_channel = equalize_channel(channel)
            channels.append(equalized_channel)
        equalized_image = cv2.merge(channels)
    else:  # Gri tonlamalı resim
        gray_image = np.clip(gray_image, 0, threshold)
        equalized_image = equalize_channel(gray_image)
    
    return equalized_image
