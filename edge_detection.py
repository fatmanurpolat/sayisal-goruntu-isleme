import numpy as np

from blurring import gaussian_blur

def sobel_edge_detection(image):
    gray = rgb2gray(image)
    sobelx = sobel_operator(gray, axis='x')
    sobely = sobel_operator(gray, axis='y')
    sobel_combined = np.sqrt(sobelx**2 + sobely**2)
    return np.uint8(sobel_combined)

def canny_edge_detection(image):
    gray = rgb2gray(image)
    blurred = gaussian_blur(gray, kernel_size=5, sigma=1.0)
    sobelx = sobel_operator(blurred, axis='x')
    sobely = sobel_operator(blurred, axis='y')
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_direction = np.arctan2(sobely, sobelx)
    non_max_suppressed = non_maximum_suppression(gradient_magnitude, gradient_direction)
    thresholded = double_threshold(non_max_suppressed, 50, 150)
    edges = hysteresis(thresholded)
    return edges

def rgb2gray(image):
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

def sobel_operator(image, axis):
    if axis == 'x':
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    else:
        kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return convolve(image, kernel)

def convolve(image, kernel):
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    output = np.zeros_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded_image[y:y + kernel_height, x:x + kernel_width]
            output[y, x] = np.sum(region * kernel)
    return output

def non_maximum_suppression(magnitude, direction):
    output = np.zeros_like(magnitude)
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            try:
                q = 255
                r = 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    output[i, j] = magnitude[i, j]
                else:
                    output[i, j] = 0

            except IndexError as e:
                pass
    return output

def double_threshold(image, low, high):
    output = np.zeros_like(image)
    strong_pixel = 255
    weak_pixel = 50

    strong_i, strong_j = np.where(image >= high)
    weak_i, weak_j = np.where((image <= high) & (image >= low))

    output[strong_i, strong_j] = strong_pixel
    output[weak_i, weak_j] = weak_pixel

    return output

def hysteresis(image):
    strong_pixel = 255
    weak_pixel = 50
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if image[i, j] == weak_pixel:
                if ((image[i + 1, j - 1] == strong_pixel) or (image[i + 1, j] == strong_pixel) or (image[i + 1, j + 1] == strong_pixel)
                    or (image[i, j - 1] == strong_pixel) or (image[i, j + 1] == strong_pixel)
                    or (image[i - 1, j - 1] == strong_pixel) or (image[i - 1, j] == strong_pixel) or (image[i - 1, j + 1] == strong_pixel)):
                    image[i, j] = strong_pixel
                else:
                    image[i, j] = 0
    return image
