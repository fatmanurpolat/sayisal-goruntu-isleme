import numpy as np

def average_blur(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    return convolve(image, kernel)

def median_blur(image, kernel_size=5):
    pad_amount = kernel_size // 2
    padded_image = np.pad(image, ((pad_amount, pad_amount), (pad_amount, pad_amount), (0, 0)), mode='constant')
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(image.shape[2]):
                region = padded_image[i:i+kernel_size, j:j+kernel_size, c]
                output[i, j, c] = np.median(region)
    return output

def gaussian_blur(image, kernel_size=5, sigma=1.0):
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve(image, kernel)

def gaussian_kernel(kernel_size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(
            -((x-(kernel_size//2))**2 + (y-(kernel_size//2))**2) / (2*sigma**2)),
        (kernel_size, kernel_size)
    )
    return kernel / np.sum(kernel)

def convolve(image, kernel):
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2
    if len(image.shape) == 3:  # Color image
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
        output = np.zeros_like(image)
        for c in range(image.shape[2]):
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    region = padded_image[y:y + kernel_height, x:x + kernel_width, c]
                    output[y, x, c] = np.sum(region * kernel)
    else:  # Grayscale image
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
        output = np.zeros_like(image)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                region = padded_image[y:y + kernel_height, x:x + kernel_width]
                output[y, x] = np.sum(region * kernel)
    return output
